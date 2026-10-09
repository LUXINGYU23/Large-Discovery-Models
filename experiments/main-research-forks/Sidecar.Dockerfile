# Build with the experiment directory as context after compiling its source.
# node_modules must match the checked-in lockfile; no moving dependency resolver.
FROM node:24.7.0-bookworm-slim
ARG DEBIAN_MIRROR=http://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-numpy qemu-system-x86 qemu-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY source/harnesses/pi/package.json source/harnesses/pi/package-lock.json ./
COPY source/harnesses/pi/node_modules ./node_modules
COPY source/harnesses/pi/dist ./dist
COPY source/harnesses/pi/policy_runner.py ./policy_runner.py
RUN chmod -R a+rX /app
USER node
ENTRYPOINT ["node", "dist/main.js"]
