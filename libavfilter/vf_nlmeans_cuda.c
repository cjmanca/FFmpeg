+/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "internal.h"

#include "cuda/load_helper.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUV444P
};


#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define BLOCKX 32
#define BLOCKY 16



typedef struct NLMeansCudaContext {
    const AVClass *class;

    double                sigma;
    int                   patch_size;
    int                   patch_size_uv;
    int                   research_size;
    int                   research_size_uv;
    int                   initialised;

    float                 h;

    AVBufferRef *hw_frames_ctx;
    AVCUDADeviceContext *hwctx;

    CUmodule    cu_module;

    CUfunction  cu_func_horiz_uchar;
    CUfunction  cu_func_horiz_uchar2;
    CUfunction  cu_func_vert_uchar;
    CUfunction  cu_func_vert_uchar2;
    CUfunction  cu_func_weight_uchar;
    CUfunction  cu_func_weight_uchar2;
    CUfunction  cu_func_average_uchar;
    CUfunction  cu_func_average_uchar2;
    CUstream    cu_stream;


    CUdeviceptr integral_img;
    CUdeviceptr integral_img2;
    CUdeviceptr weight;
    CUdeviceptr weight2;
    CUdeviceptr sum;
    CUdeviceptr sum2;

} NLMeansCudaContext;

#define OFFSET(x) offsetof(NLMeansCudaContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption nlmeans_cuda_options[] = {
    { "s",  "denoising strength", OFFSET(sigma), AV_OPT_TYPE_DOUBLE, { .dbl = 1.0 }, 1.0, 30.0, FLAGS },
    { "p",  "patch size",                   OFFSET(patch_size),    AV_OPT_TYPE_INT, { .i64 = 2*3+1 }, 0, 99, FLAGS },
    { "pc", "patch size for chroma planes", OFFSET(patch_size_uv), AV_OPT_TYPE_INT, { .i64 = 0 },     0, 99, FLAGS },
    { "r",  "research window",                   OFFSET(research_size),    AV_OPT_TYPE_INT, { .i64 = 7*2+1 }, 0, 99, FLAGS },
    { "rc", "research window for chroma planes", OFFSET(research_size_uv), AV_OPT_TYPE_INT, { .i64 = 0 },     0, 99, FLAGS },
    { NULL }
};



static av_cold int init(AVFilterContext *avctx)
{
    NLMeansCudaContext *ctx = avctx->priv;

    ctx->h = ctx->sigma * 10;
    if (!(ctx->research_size & 1)) {
        ctx->research_size |= 1;
        av_log(avctx, AV_LOG_WARNING,
               "research_size should be odd, set to %d",
               ctx->research_size);
    }

    if (!(ctx->patch_size & 1)) {
        ctx->patch_size |= 1;
        av_log(avctx, AV_LOG_WARNING,
               "patch_size should be odd, set to %d",
               ctx->patch_size);
    }

    if (!ctx->research_size_uv)
        ctx->research_size_uv = ctx->research_size;
    if (!ctx->patch_size_uv)
        ctx->patch_size_uv = ctx->patch_size;


    ctx->initialised = 1;

    return 0;
}


static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}


static int call_average2(AVFilterContext *ctx, int channels,
                         uint8_t *src_dptr, int src_width, int src_height, int src_pitch,
                         float *sum, float *sum2, float *weight, float *weight2,
                         uint8_t *dst_dptr, int dst_width, int dst_height, int dst_pitch,
                         int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    CUtexObject tex = 0;
    void *args_uchar[] = { &tex, &dst_devptr, &src_width, &dst_pitch, &sum, &sum2, &weight, &weight2};

    int ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    dst_pitch /= channels * pixel_size;

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_average_uchar2,
                                      DIV_UP(dst_width, BLOCKX), DIV_UP(dst_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}


static int call_average(AVFilterContext *ctx, int channels,
                        uint8_t *src_dptr, int src_width, int src_height, int src_pitch, float *sum, float *weight,
                        uint8_t *dst_dptr, int dst_width, int dst_height, int dst_pitch,
                        int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    CUtexObject tex = 0;
    void *args_uchar[] = { &tex, &dst_devptr, &src_width, &dst_pitch, &sum, &weight};

    int ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    dst_pitch /= channels * pixel_size;

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_average_uchar,
                                      DIV_UP(dst_width, BLOCKX), DIV_UP(dst_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}

static int call_weight2(AVFilterContext *ctx, int channels,
                        uint8_t *src_dptr, int src_width, int src_height, int src_pitch, int *integ_img, int *integ_img2,
                        float *sum, float *sum2, float *weight, float *weight2, int p,
                        int *dx_cur, int *dy_cur,
                        int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    CUtexObject tex = 0;
    int ret;


    void *args_uchar[] = { &sum, &sum2, &weight, &weight2, &integ_img, &integ_img2, &tex, &src_width, &src_height, &p, &s->h, dx_cur, dy_cur };


    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    src_pitch /= channels * pixel_size;

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_weight_uchar2,
                                      DIV_UP(src_width, BLOCKX), DIV_UP(src_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}



static int call_weight(AVFilterContext *ctx, int channels,
                       uint8_t *src_dptr, int src_width, int src_height, int src_pitch, int *integ_img, float *sum, float *weight, int p,
                       int *dx_cur, int *dy_cur,
                       int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    CUtexObject tex = 0;
    int ret;


    void *args_uchar[] = { &sum, &weight, &integ_img, &tex, &src_width, &src_height, &p, &s->h, dx_cur, dy_cur };


    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    src_pitch /= channels * pixel_size;

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_weight_uchar,
                                      DIV_UP(src_width, BLOCKX), DIV_UP(src_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}


static int call_vert2(AVFilterContext *ctx, int channels,
                      int src_width, int src_height, int *integ_img, int *integ_img2,
                      int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    int ret;

    void *args_uchar[] = { &integ_img, &integ_img2, &src_width, &src_height };


    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_vert_uchar2,
                                      DIV_UP(src_width, BLOCKX), 1, 1,
                                      BLOCKX, 1, 1, 0, s->cu_stream, args_uchar, NULL));


    return ret;
}


static int call_vert(AVFilterContext *ctx, int channels,
                     int src_width, int src_height, int *integ_img,
                     int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    int ret;

    void *args_uchar[] = { &integ_img, &src_width, &src_height };


    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_vert_uchar,
                                      DIV_UP(src_width, BLOCKX), 1, 1,
                                      BLOCKX, 1, 1, 0, s->cu_stream, args_uchar, NULL));



    return ret;
}


static int call_horiz2(AVFilterContext *ctx, int channels,
                       uint8_t *src_dptr, int src_width, int src_height, int src_pitch, int *integ_img, int *integ_img2,
                       int *dx_cur, int *dy_cur,
                       int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    CUtexObject tex = 0;
    int ret;


    void *args_uchar[] = { &integ_img, &integ_img2, &tex, &src_width, &src_height, dx_cur, dy_cur };


    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    src_pitch /= channels * pixel_size;

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_horiz_uchar2,
                                      1, DIV_UP(src_height, BLOCKY), 1,
                                      1, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}


static int call_horiz(AVFilterContext *ctx, int channels,
                      uint8_t *src_dptr, int src_width, int src_height, int src_pitch, int *integ_img,
                      int *dx_cur, int *dy_cur,
                      int pixel_size)
{
    NLMeansCudaContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    CUtexObject tex = 0;
    int ret;


    void *args_uchar[] = { &integ_img, &tex, &src_width, &src_height, dx_cur, dy_cur };


    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_POINT,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = channels,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    src_pitch /= channels * pixel_size;

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_horiz_uchar,
                                      1, DIV_UP(src_height, BLOCKY), 1,
                                      1, BLOCKY, 1, 0, s->cu_stream, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}


static void nlmeans_plane(AVFilterContext *ctx, int channels,
                          uint8_t *src_dptr, int src_width, int src_height, int src_pitch,
                          uint8_t *dst_dptr, int dst_width, int dst_height, int dst_pitch,
                          int *integ_img,
                          int p, int r,
                          int pixel_size)
{
    NLMeansCudaContext *s   = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;


    int nb_pixel, *tmp = NULL, idx = 0;
    int *dxdy = NULL;
    int i, dx, dy = 0;

    nb_pixel = (2 * r + 1) * (2 * r + 1) - 1;
    dxdy = av_malloc(nb_pixel * 2 * sizeof(int));
    tmp = av_malloc(nb_pixel * 2 * sizeof(int));

    for (dx = -r; dx <= r; dx++) {
        for (dy = -r; dy <= r; dy++) {
            if (dx || dy) {
                tmp[idx++] = dx;
                tmp[idx++] = dy;
            }
        }
    }
    // repack dx/dy seperately, as we want to do four pairs of dx/dy in a batch
    for (i = 0; i < nb_pixel / 4; i++) {
        dxdy[i * 8] = tmp[i * 8];         // dx0
        dxdy[i * 8 + 1] = tmp[i * 8 + 2]; // dx1
        dxdy[i * 8 + 2] = tmp[i * 8 + 4]; // dx2
        dxdy[i * 8 + 3] = tmp[i * 8 + 6]; // dx3
        dxdy[i * 8 + 4] = tmp[i * 8 + 1]; // dy0
        dxdy[i * 8 + 5] = tmp[i * 8 + 3]; // dy1
        dxdy[i * 8 + 6] = tmp[i * 8 + 5]; // dy2
        dxdy[i * 8 + 7] = tmp[i * 8 + 7]; // dy3
    }
    av_freep(&tmp);


    // fill with 0s
    CHECK_CU(cu->cuMemsetD8Async(s->weight, 0, src_width * src_height * sizeof(float), s->cu_stream));
    CHECK_CU(cu->cuMemsetD8Async(s->sum, 0, src_width * src_height * sizeof(float), s->cu_stream));


    for (i = 0; i < nb_pixel / 4; i++) {

        int *dx_cur = dxdy + 8 * i;
        int *dy_cur = dxdy + 8 * i + 4;

        call_horiz(ctx, 1, src_dptr, src_width, src_height, src_pitch,
                   integ_img, dx_cur, dy_cur, pixel_size);

        call_vert(ctx, 1, src_width, src_height, integ_img, pixel_size);

        call_weight(ctx, 1, src_dptr, src_width, src_height, src_pitch, integ_img, (float*)s->sum, (float*)s->weight, p, dx_cur, dy_cur, pixel_size);
    }

    call_average(ctx, 1, src_dptr, src_width, src_height, src_pitch, (float*)s->sum, (float*)s->weight,
                   dst_dptr, dst_width, dst_height, dst_pitch, pixel_size);
}

static void nlmeans_plane2(AVFilterContext *ctx, int channels,
                           uint8_t *src_dptr, int src_width, int src_height, int src_pitch,
                           uint8_t *dst_dptr, int dst_width, int dst_height, int dst_pitch,
                           int *integ_img, int *integ_img2,
                           int p, int r,
                           int pixel_size)
{
    NLMeansCudaContext *s   = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    int nb_pixel, *tmp = NULL, idx = 0;
    int *dxdy = NULL;
    int i, dx, dy = 0;

    nb_pixel = (2 * r + 1) * (2 * r + 1) - 1;
    dxdy = av_malloc(nb_pixel * 2 * sizeof(int));
    tmp = av_malloc(nb_pixel * 2 * sizeof(int));

    for (dx = -r; dx <= r; dx++) {
        for (dy = -r; dy <= r; dy++) {
            if (dx || dy) {
                tmp[idx++] = dx;
                tmp[idx++] = dy;
            }
        }
    }
    // repack dx/dy seperately, as we want to do four pairs of dx/dy in a batch
    for (i = 0; i < nb_pixel / 4; i++) {
        dxdy[i * 8] = tmp[i * 8];         // dx0
        dxdy[i * 8 + 1] = tmp[i * 8 + 2]; // dx1
        dxdy[i * 8 + 2] = tmp[i * 8 + 4]; // dx2
        dxdy[i * 8 + 3] = tmp[i * 8 + 6]; // dx3
        dxdy[i * 8 + 4] = tmp[i * 8 + 1]; // dy0
        dxdy[i * 8 + 5] = tmp[i * 8 + 3]; // dy1
        dxdy[i * 8 + 6] = tmp[i * 8 + 5]; // dy2
        dxdy[i * 8 + 7] = tmp[i * 8 + 7]; // dy3
    }
    av_freep(&tmp);


    // fill with 0s
    CHECK_CU(cu->cuMemsetD8Async(s->weight, 0, src_width * src_height * sizeof(float), s->cu_stream));
    CHECK_CU(cu->cuMemsetD8Async(s->sum, 0, src_width * src_height * sizeof(float), s->cu_stream));
    CHECK_CU(cu->cuMemsetD8Async(s->weight2, 0, src_width * src_height * sizeof(float), s->cu_stream));
    CHECK_CU(cu->cuMemsetD8Async(s->sum2, 0, src_width * src_height * sizeof(float), s->cu_stream));

    for (i = 0; i < nb_pixel / 4; i++) {

        int *dx_cur = dxdy + 8 * i;
        int *dy_cur = dxdy + 8 * i + 4;

        call_horiz2(ctx, 2, src_dptr, src_width, src_height, src_pitch,
                    integ_img, integ_img2, dx_cur, dy_cur, pixel_size);

        call_vert2(ctx, 2, src_width, src_height, integ_img, integ_img2, pixel_size);

        call_weight2(ctx, 2, src_dptr, src_width, src_height, src_pitch, integ_img, integ_img2, (float*)s->sum, (float*)s->sum2, (float*)s->weight, (float*)s->weight2, p, dx_cur, dy_cur, pixel_size);
    }

    call_average2(ctx, 2, src_dptr, src_width, src_height, src_pitch, (float*)s->sum, (float*)s->sum2, (float*)s->weight, (float*)s->weight2,
                  dst_dptr, dst_width, dst_height, dst_pitch, pixel_size);
}


static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx  = inlink->dst;
    NLMeansCudaContext *s   = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    AVFilterLink *outlink = ctx->outputs[0];
    AVHWFramesContext *hw_frames_ctx = (AVHWFramesContext*)s->hw_frames_ctx->data;
    CUcontext dummy;

    AVFrame *output = NULL;
    int err, patch, research, patch_uv, research_uv;

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    err = av_frame_copy_props(output, frame);
    if (err < 0)
        goto fail;

    if (!s->initialised) {
        init(ctx);
    }

    patch = s->patch_size / 2;
    research = s->research_size / 2;
    patch_uv = s->patch_size_uv / 2;
    research_uv = s->research_size_uv / 2;

    err = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (err < 0)
        return err;



    switch (hw_frames_ctx->sw_format) {
    case AV_PIX_FMT_NV12:
        nlmeans_plane(ctx, 1, frame->data[0], inlink->w, inlink->h, frame->linesize[0],
                      output->data[0], output->width, output->height, output->linesize[0],
                      (int*)s->integral_img, patch, research, 1);

        nlmeans_plane2(ctx, 2, frame->data[1], inlink->w / 2, inlink->h / 2, frame->linesize[1],
                       output->data[1], output->width / 2, output->height / 2, output->linesize[1],
                       (int*)s->integral_img, (int*)s->integral_img2, patch_uv, research_uv, 1);

        break;
    case AV_PIX_FMT_YUV420P:
        nlmeans_plane(ctx, 1, frame->data[0], inlink->w, inlink->h, frame->linesize[0],
                      output->data[0], output->width, output->height, output->linesize[0],
                      (int*)s->integral_img, patch, research, 1);

        nlmeans_plane(ctx, 1, frame->data[1], inlink->w / 2, inlink->h / 2, frame->linesize[1],
                      output->data[1], output->width / 2, output->height / 2, output->linesize[1],
                      (int*)s->integral_img, patch_uv, research_uv, 1);

        nlmeans_plane(ctx, 1, frame->data[2], inlink->w / 2, inlink->h / 2, frame->linesize[2],
                      output->data[2], output->width / 2, output->height / 2, output->linesize[2],
                      (int*)s->integral_img, patch_uv, research_uv, 1);

        break;
    case AV_PIX_FMT_YUV444P:
        nlmeans_plane(ctx, 1, frame->data[0], inlink->w, inlink->h, frame->linesize[0],
                      output->data[0], output->width, output->height, output->linesize[0],
                      (int*)s->integral_img, patch, research, 1);

        nlmeans_plane(ctx, 1, frame->data[1], inlink->w, inlink->h, frame->linesize[1],
                      output->data[1], output->width, output->height, output->linesize[1],
                      (int*)s->integral_img, patch_uv, research_uv, 1);

        nlmeans_plane(ctx, 1, frame->data[2], inlink->w, inlink->h, frame->linesize[2],
                      output->data[2], output->width, output->height, output->linesize[2],
                      (int*)s->integral_img, patch_uv, research_uv, 1);

        break;
    default:
        return AVERROR_BUG;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (err < 0)
        return err;

    av_frame_free(&frame);

    return ff_filter_frame(outlink, output);

fail:
    av_frame_free(&frame);
    av_frame_free(&output);
    return err;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    NLMeansCudaContext *s = ctx->priv;

    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;

        if (s->integral_img) {
            CHECK_CU(cu->cuMemFree(s->integral_img));
            s->integral_img = 0;
        }

        if (s->integral_img2) {
            CHECK_CU(cu->cuMemFree(s->integral_img2));
            s->integral_img2 = 0;
        }

        if (s->weight) {
            CHECK_CU(cu->cuMemFree(s->weight));
            s->weight = 0;
        }

        if (s->weight2) {
            CHECK_CU(cu->cuMemFree(s->weight2));
            s->weight2 = 0;
        }

        if (s->sum) {
            CHECK_CU(cu->cuMemFree(s->sum));
            s->sum = 0;
        }

        if (s->sum2) {
            CHECK_CU(cu->cuMemFree(s->sum2));
            s->sum2 = 0;
        }

        if (s->cu_module) {
            CHECK_CU(cu->cuModuleUnload(s->cu_module));
            s->cu_module = NULL;
        }
    }
}


static int config_props(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    NLMeansCudaContext *s = ctx->priv;
    AVHWFramesContext     *hw_frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = hw_frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CudaFunctions *cu = device_hwctx->internal->cuda_dl;
    int ret;

    extern const unsigned char ff_vf_nlmeans_cuda_ptx_data[];
    extern const unsigned int ff_vf_nlmeans_cuda_ptx_len;

    int width = inlink->w;
    int height = inlink->h;

    s->hwctx = device_hwctx;
    s->cu_stream = s->hwctx->stream;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    ret = ff_cuda_load_module(ctx, device_hwctx, &s->cu_module, ff_vf_nlmeans_cuda_ptx_data, ff_vf_nlmeans_cuda_ptx_len);
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_horiz_uchar, s->cu_module, "nlmeans_horiz_uchar"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_horiz_uchar2, s->cu_module, "nlmeans_horiz_uchar2"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_vert_uchar, s->cu_module, "nlmeans_vert_uchar"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_vert_uchar2, s->cu_module, "nlmeans_vert_uchar2"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_weight_uchar, s->cu_module, "nlmeans_weight_uchar"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_weight_uchar2, s->cu_module, "nlmeans_weight_uchar2"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_average_uchar, s->cu_module, "nlmeans_average_uchar"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_average_uchar2, s->cu_module, "nlmeans_average_uchar2"));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuMemAlloc(&s->integral_img, 4 * width * height * sizeof(int)));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuMemAlloc(&s->integral_img2, 4 * width * height * sizeof(int)));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuMemAlloc(&s->weight, width * height * sizeof(float)));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuMemAlloc(&s->sum, width * height * sizeof(float)));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuMemAlloc(&s->weight2, width * height * sizeof(float)));
    if (ret < 0)
        return ret;

    ret = CHECK_CU(cu->cuMemAlloc(&s->sum2, width * height * sizeof(float)));
    if (ret < 0)
        return ret;

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    s->hw_frames_ctx = ctx->inputs[0]->hw_frames_ctx;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->hw_frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);


    if (!format_is_supported(hw_frames_ctx->sw_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n", av_get_pix_fmt_name(hw_frames_ctx->sw_format));
        return AVERROR(ENOSYS);
    }

    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_CUDA,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

AVFILTER_DEFINE_CLASS(nlmeans_cuda);

static const AVFilterPad nlmeans_cuda_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
        .filter_frame = filter_frame,
    }
};

static const AVFilterPad nlmeans_cuda_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
    }
};

const AVFilter ff_vf_nlmeans_cuda = {
    .name          = "nlmeans_cuda",
    .description   = NULL_IF_CONFIG_SMALL("Non-local means denoiser through CUDA"),
    .priv_size     = sizeof(NLMeansCudaContext),
    .init          = init,
    .uninit        = uninit,
    FILTER_INPUTS(nlmeans_cuda_inputs),
    FILTER_OUTPUTS(nlmeans_cuda_outputs),
    FILTER_QUERY_FUNC(query_formats),
    .priv_class    = &nlmeans_cuda_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};