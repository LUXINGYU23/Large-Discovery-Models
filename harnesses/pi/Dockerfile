FROM node:24.7.0-bookworm-slim AS dependencies

ARG NPM_REGISTRY=https://registry.npmjs.org
WORKDIR /app
COPY package.json package-lock.json ./
COPY resolve-sol-pi.mjs ./
# ADD revalidates this remote revision on each build; the image records its exact pair.
ADD https://api.github.com/repos/NVlabs/SoL-Pi/commits/main /tmp/sol-pi-head.json
RUN node resolve-sol-pi.mjs /tmp/sol-pi-head.json ${NPM_REGISTRY}
RUN npm ci --registry=${NPM_REGISTRY}

FROM dependencies AS build

COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM dependencies AS production-dependencies

RUN npm prune --omit=dev --no-audit --no-fund --offline

FROM node:24.7.0-bookworm-slim

ARG DEBIAN_MIRROR=http://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-numpy qemu-system-x86 qemu-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=production-dependencies /app/package.json /app/package-lock.json ./
COPY --from=production-dependencies /app/sol-pi-version.json ./
COPY --from=production-dependencies /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY policy_runner.py ./policy_runner.py
RUN chmod 0644 package.json package-lock.json policy_runner.py

USER node
ENTRYPOINT ["node", "dist/main.js"]
