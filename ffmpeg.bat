rem git clone https://git.ffmpeg.org/ffmpeg.git
rem cd ffmpeg
rem wget https://patchwork.ffmpeg.org/project/ffmpeg/patch/CAPa-kAzCkbAmOVoyD2__HB2gXAFDz8oLyBJ7UAwyqu-Nahy8Nw@mail.gmail.com/raw/ -O nlmeans-cuda.diff
git apply nlmeans-cuda.diff
rem ./configure
rem make
rem make install
pause