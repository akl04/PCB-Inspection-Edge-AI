#!/bin/bash

URL="http://127.0.0.1:7860/"

for ((i=1; i<=120; i++))
do
    if curl --noproxy '*' -fsS \
        --max-time 2 \
        -o /dev/null "$URL"
    then
        exec /usr/bin/chromium \
            --no-proxy-server \
            --disable-gpu \
            --disable-extensions \
            --disable-background-networking \
            --no-first-run \
            --no-default-browser-check \
            --user-data-dir="$HOME/.config/pcb-standalone-browser" \
            --start-maximized \
            --app="$URL"
    fi

    sleep 1
done

echo "Dashboard unavailable."
exit 1
