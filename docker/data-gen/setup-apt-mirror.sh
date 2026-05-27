#!/bin/sh
set -e
rm -f /etc/apt/sources.list.d/ubuntu.sources
. /etc/os-release
exec >/etc/apt/sources.list
echo "deb ${APT_MIRROR} ${VERSION_CODENAME} main restricted universe multiverse"
echo "deb ${APT_MIRROR} ${VERSION_CODENAME}-updates main restricted universe multiverse"
echo "deb ${APT_MIRROR} ${VERSION_CODENAME}-security main restricted universe multiverse"
echo "deb ${APT_MIRROR} ${VERSION_CODENAME}-backports main restricted universe multiverse"
