#!/bin/bash
# Google Cloud-ის მანქანაზე (systemd: aivoice-autostop.service, setup_vm.sh აყენებს):
# წვრთნა დასრულდა (/data/DONE ≥ 30 წთ), მანქანა ≥ 60 წთ ჩართულია და წვრთნა არ მიდის → გათიშვა.
# გათიშული მანქანა GPU-ს არ იხდის; ჩვენი მეთვალყურე (tools/gcp_train.py watch) მას ჩართავს,
# შედეგს ჩამოტვირთავს და წაშლის — ჩართვიდან 60 წთ-ის განმავლობაში აქ არაფერი ხდება.
while sleep 300; do
  up=$(cut -d. -f1 /proc/uptime)
  if [ -f /data/DONE ] && [ $(( $(date +%s) - $(stat -c %Y /data/DONE) )) -gt 1800 ] && [ "$up" -gt 3600 ] \
     && ! systemctl is-active -q aivoice-train; then
    echo "$(date -Is) autostop" >> /data/train.log
    shutdown -h now
  fi
done
