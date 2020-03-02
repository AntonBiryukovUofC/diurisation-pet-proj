#!/bin/bash



docker run -it -p 5000:5000 -p 5006:5006 --device /dev/snd \
-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
-v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
--group-add $(getent group audio | cut -d: -f3) --device /dev/snd celeb-recog:latest