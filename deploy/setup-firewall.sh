#!/bin/bash
# Server firewall setup — run once on the Hetzner host after initial provisioning.
# Usage: sudo bash deploy/setup-firewall.sh
set -euo pipefail

echo "=== Setting up UFW firewall ==="
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw --force enable
echo "UFW enabled."

echo "=== Blocking outbound UDP on common attack ports ==="
iptables -A OUTPUT -p udp --dport 80 -j DROP
iptables -A OUTPUT -p udp --dport 443 -j DROP
iptables -A OUTPUT -p udp --dport 8080 -j DROP
echo "UDP rules added."

echo "=== Persisting iptables rules ==="
DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
netfilter-persistent save
echo "Rules saved."

echo "=== Firewall setup complete ==="
ufw status verbose
iptables -L OUTPUT -n --line-numbers
