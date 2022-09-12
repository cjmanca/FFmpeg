/*
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
#include "cuda/vector_helpers.cuh"


extern "C" {

__global__ void nlmeans_average_uchar(
    cudaTextureObject_t tex,
    uchar *dst,
    int width, int dst_pitch,
    float* sum, float* weight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float w = weight[y * width + x];
    float s = sum[y * width + x];

    uchar src_pix_uchar = tex2D<uchar>(tex, x, y);
    int src_pix_int = src_pix_uchar;
    float src_pix = (float)src_pix_int;

    float resultf = (s + src_pix) / (1 + w);

    int result = (int)resultf;

    dst[y*dst_pitch+x] = result;
}


__global__ void nlmeans_average_uchar2(
    cudaTextureObject_t tex,
    uchar2 *dst,
    int width, int dst_pitch,
    float* sum, float* sum2, float* weight, float* weight2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float w = weight[y * width + x];
    float s = sum[y * width + x];

    float w2 = weight2[y * width + x];
    float s2 = sum2[y * width + x];

    uchar2 src_pix_uchar = tex2D<uchar2>(tex, x, y);

    int src_pix_int = src_pix_uchar.x;
    float src_pix = (float)src_pix_int;

    int src_pix_int2 = src_pix_uchar.y;
    float src_pix2 = (float)src_pix_int2;

    float resultf = (s + src_pix) / (1 + w);
    float result2f = (s2 + src_pix2) / (1 + w2);

    int result = (int)resultf;
    int result2 = (int)result2f;

    uchar2 output;
    output.x = result;
    output.y = result2;

    dst[y*dst_pitch+x] = output;
}


__global__ void nlmeans_weight_uchar2(
    float* sum, float* sum2,
    float* weight, float* weight2,
    int4 *integral_img,
    int4 *integral_img2,
    cudaTextureObject_t tex,
    int width, int height,
    int p, float h,
    int4 dx, int4 dy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int4 xoff = make_int4(x, x, x, x) + dx;
    int4 yoff = make_int4(y, y, y, y) + dy;

    int4 a = make_int4(0, 0, 0, 0);
    int4 b = make_int4(0, 0, 0, 0);
    int4 c = make_int4(0, 0, 0, 0);
    int4 d = make_int4(0, 0, 0, 0);

    int4 a2 = make_int4(0, 0, 0, 0);
    int4 b2 = make_int4(0, 0, 0, 0);
    int4 c2 = make_int4(0, 0, 0, 0);
    int4 d2 = make_int4(0, 0, 0, 0);

    int4 src_pix  = make_int4(0, 0, 0, 0);
    int4 src_pix2 = make_int4(0, 0, 0, 0);

    int oobb = (x - p) < 0 || (y - p) < 0 || (y + p) >= height || (x + p) >= width;

    uchar2 ax, ay, az, aw;
    ax = tex2D<uchar2>(tex, xoff.x, yoff.x);
    ay = tex2D<uchar2>(tex, xoff.y, yoff.y);
    az = tex2D<uchar2>(tex, xoff.z, yoff.z);
    aw = tex2D<uchar2>(tex, xoff.w, yoff.w);

    src_pix.x = ax.x;
    src_pix.y = ay.x;
    src_pix.z = az.x;
    src_pix.w = aw.x;

    src_pix2.x = ax.y;
    src_pix2.y = ay.y;
    src_pix2.z = az.y;
    src_pix2.w = aw.y;

    if (!oobb) {
        a = integral_img[(y - p) * width + x - p];
        b = integral_img[(y + p) * width + x - p];
        c = integral_img[(y - p) * width + x + p];
        d = integral_img[(y + p) * width + x + p];

        a2 = integral_img2[(y - p) * width + x - p];
        b2 = integral_img2[(y + p) * width + x - p];
        c2 = integral_img2[(y - p) * width + x + p];
        d2 = integral_img2[(y + p) * width + x + p];
    }

    int4 patch_diff_int = d + a - c - b;
    float4 patch_diff;
    patch_diff.x = (float) (-patch_diff_int.x);
    patch_diff.y = (float) (-patch_diff_int.y);
    patch_diff.z = (float) (-patch_diff_int.z);
    patch_diff.w = (float) (-patch_diff_int.w);

    int4 patch_diff_int2 = d2 + a2 - c2 - b2;
    float4 patch_diff2;
    patch_diff2.x = (float) (-patch_diff_int2.x);
    patch_diff2.y = (float) (-patch_diff_int2.y);
    patch_diff2.z = (float) (-patch_diff_int2.z);
    patch_diff2.w = (float) (-patch_diff_int2.w);


    float4 w;
    w.x = __expf(patch_diff.x / (h*h));
    w.y = __expf(patch_diff.y / (h*h));
    w.z = __expf(patch_diff.z / (h*h));
    w.w = __expf(patch_diff.w / (h*h));

    float4 w2;
    w2.x = __expf(patch_diff2.x / (h*h));
    w2.y = __expf(patch_diff2.y / (h*h));
    w2.z = __expf(patch_diff2.z / (h*h));
    w2.w = __expf(patch_diff2.w / (h*h));

    float w_sum = w.x + w.y + w.z + w.w;
    weight[y * width + x] += w_sum;

    float w_sum2 = w2.x + w2.y + w2.z + w2.w;
    weight2[y * width + x] += w_sum2;

    float4 src_pixf;
    src_pixf.x = (float)src_pix.x;
    src_pixf.y = (float)src_pix.y;
    src_pixf.z = (float)src_pix.z;
    src_pixf.w = (float)src_pix.w;

    float4 src_pix2f;
    src_pix2f.x = (float)src_pix2.x;
    src_pix2f.y = (float)src_pix2.y;
    src_pix2f.z = (float)src_pix2.z;
    src_pix2f.w = (float)src_pix2.w;

    sum[y * width + x] += w.x * src_pixf.x + w.y * src_pixf.y + w.z * src_pixf.z + w.w * src_pixf.w;
    sum2[y * width + x] += w2.x * src_pix2f.x + w2.y * src_pix2f.y + w2.z * src_pix2f.z + w2.w * src_pix2f.w;
}


__global__ void nlmeans_weight_uchar(
    float* sum, float* weight,
    int4 *integral_img,
    cudaTextureObject_t tex,
    int width, int height,
    int p, float h,
    int4 dx, int4 dy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int4 xoff = make_int4(x, x, x, x) + dx;
    int4 yoff = make_int4(y, y, y, y) + dy;

    int4 a = make_int4(0, 0, 0, 0);
    int4 b = make_int4(0, 0, 0, 0);
    int4 c = make_int4(0, 0, 0, 0);
    int4 d = make_int4(0, 0, 0, 0);

    int4 src_pix = make_int4(0, 0, 0, 0);

    int oobb = (x - p) < 0 || (y - p) < 0 || (y + p) >= height || (x + p) >= width;

    src_pix.x = tex2D<uchar>(tex, xoff.x, yoff.x);
    src_pix.y = tex2D<uchar>(tex, xoff.y, yoff.y);
    src_pix.z = tex2D<uchar>(tex, xoff.z, yoff.z);
    src_pix.w = tex2D<uchar>(tex, xoff.w, yoff.w);

    if (!oobb) {
        a = integral_img[(y - p) * width + x - p];
        b = integral_img[(y + p) * width + x - p];
        c = integral_img[(y - p) * width + x + p];
        d = integral_img[(y + p) * width + x + p];
    }

    int4 patch_diff_int = d + a - c - b;
    float4 patch_diff;
    patch_diff.x = (float) (-patch_diff_int.x);
    patch_diff.y = (float) (-patch_diff_int.y);
    patch_diff.z = (float) (-patch_diff_int.z);
    patch_diff.w = (float) (-patch_diff_int.w);


    float4 w;
    w.x = __expf(patch_diff.x / (h*h));
    w.y = __expf(patch_diff.y / (h*h));
    w.z = __expf(patch_diff.z / (h*h));
    w.w = __expf(patch_diff.w / (h*h));

    float w_sum = w.x + w.y + w.z + w.w;
    weight[y * width + x] += w_sum;

    float4 src_pixf;
    src_pixf.x = (float)src_pix.x;
    src_pixf.y = (float)src_pix.y;
    src_pixf.z = (float)src_pix.z;
    src_pixf.w = (float)src_pix.w;

    sum[y * width + x] += w.x * src_pixf.x + w.y * src_pixf.y + w.z * src_pixf.z + w.w * src_pixf.w;
}


__global__ void nlmeans_vert_uchar(
    int4 *integral_img,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int4 sum = make_int4(0, 0, 0, 0);

    for (int i = 0; i < height; i++) {
        integral_img[i * width + x] += sum;
        sum = integral_img[i * width + x];
    }
}


__global__ void nlmeans_vert_uchar2(
    int4 *integral_img,
    int4 *integral_img2,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int4 sum = make_int4(0, 0, 0, 0);
    int4 sum2 = make_int4(0, 0, 0, 0);

    for (int i = 0; i < height; i++) {
        integral_img[i * width + x] += sum;
        sum = integral_img[i * width + x];

        integral_img2[i * width + x] += sum2;
        sum2 = integral_img2[i * width + x];
    }
}


__global__ void nlmeans_horiz_uchar(
    int4 *integral_img,
    cudaTextureObject_t tex,
    int width, int height,
    int4 dx, int4 dy)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int4 sum = make_int4(0, 0, 0, 0);
    uchar s2x, s2y, s2z, s2w;

    for (int i = 0; i < width; i++) {
        uchar s1c = tex2D<uchar>(tex, i, y);
        int4 s1 = make_int4(s1c, s1c, s1c, s1c);

        s2x = tex2D<uchar>(tex, i + dx.x, y + dy.x);
        s2y = tex2D<uchar>(tex, i + dx.y, y + dy.y);
        s2z = tex2D<uchar>(tex, i + dx.z, y + dy.z);
        s2w = tex2D<uchar>(tex, i + dx.w, y + dy.w);

        int4 s2 = make_int4(s2x, s2y, s2z, s2w);

        //sum += (s1 - s2) * (s1 - s2);
        int4 s3 = s1 - s2;
        int4 s4;
        s4.x = s3.x * s3.x;
        s4.y = s3.y * s3.y;
        s4.z = s3.z * s3.z;
        s4.w = s3.w * s3.w;

        sum += s4;

        integral_img[y * width + i] = sum;
    }
}


__global__ void nlmeans_horiz_uchar2(
    int4 *integral_img,
    int4 *integral_img2,
    cudaTextureObject_t tex,
    int width, int height,
    int4 dx, int4 dy)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int4 sum = make_int4(0, 0, 0, 0);
    int4 sum2 = make_int4(0, 0, 0, 0);
    uchar2 s2x, s2y, s2z, s2w;

    for (int i = 0; i < width; i++) {
        uchar2 s1c = tex2D<uchar2>(tex, i, y);
        int4 s1 = make_int4(s1c.x, s1c.x, s1c.x, s1c.x);
        int4 r1 = make_int4(s1c.y, s1c.y, s1c.y, s1c.y);

        s2x = tex2D<uchar2>(tex, i + dx.x, y + dy.x);
        s2y = tex2D<uchar2>(tex, i + dx.y, y + dy.y);
        s2z = tex2D<uchar2>(tex, i + dx.z, y + dy.z);
        s2w = tex2D<uchar2>(tex, i + dx.w, y + dy.w);

        int4 s2 = make_int4(s2x.x, s2y.x, s2z.x, s2w.x);
        int4 r2 = make_int4(s2x.y, s2y.y, s2z.y, s2w.y);

        //sum += (s1 - s2) * (s1 - s2);
        int4 s3 = s1 - s2;
        int4 s4;
        s4.x = s3.x * s3.x;
        s4.y = s3.y * s3.y;
        s4.z = s3.z * s3.z;
        s4.w = s3.w * s3.w;

        sum += s4;

        //sum2 += (r1 - r2) * (r1 - r2);
        int4 r3 = r1 - r2;
        int4 r4;
        r4.x = r3.x * r3.x;
        r4.y = r3.y * r3.y;
        r4.z = r3.z * r3.z;
        r4.w = r3.w * r3.w;

        sum2 += r4;

        integral_img[y * width + i] = sum;
        integral_img2[y * width + i] = sum2;
    }
}

}
