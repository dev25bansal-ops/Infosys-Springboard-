# MLOPS-17: the Binance stream + Socket.io service (mini-services/binance-stream).
FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps || npm install --legacy-peer-deps
COPY index.ts outbox.ts env.ts correlation.ts ./
ENV HEALTH_PORT=3005
EXPOSE 3003 3005
CMD ["npx", "tsx", "index.ts"]