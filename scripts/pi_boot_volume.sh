#!/usr/bin/env bash
# Set the Pi's speaker volume at boot (deploy/vcm-volume.service): the real output sinks (Bluetooth speaker, built-in
# jack) go to VOL; the echo-cancel virtual sink stays at 100 % so the volume is not applied twice. A Bluetooth speaker
# can connect a while after boot, so the script keeps watching for up to WAIT seconds and sets each new sink once.
#   bash scripts/pi_boot_volume.sh [VOL=100%] [WAIT=180]
VOL="${1:-100%}"
WAIT="${2:-180}"
declare -A done_
end=$((SECONDS + WAIT))
while [ $SECONDS -lt $end ]; do
  while read -r _ name _; do
    [ -z "$name" ] || [ -n "${done_[$name]:-}" ] && continue
    if [ "$name" = "echo-cancel-sink" ]; then
      pactl set-sink-volume "$name" 100% && done_[$name]=1
    else
      pactl set-sink-volume "$name" "$VOL" && done_[$name]=1 && echo "set $name to $VOL"
    fi
  done < <(pactl list short sinks 2>/dev/null)
  sleep 3
done
