# Builds layad for linux/amd64 or linux/arm64. Mount the exported checkpoints at /models.
#
#   docker build -t layad .
#   docker run --rm -p 8080:8080 -v $PWD/models:/models:ro layad
FROM golang:1.25-bookworm AS build
ARG ORT_VERSION=1.29.1
ARG TOKENIZERS_VERSION=1.27.0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN ORT_VERSION=$ORT_VERSION TOKENIZERS_VERSION=$TOKENIZERS_VERSION scripts/fetch_deps.sh \
 && . third_party/env.sh && go build -o /out/layad ./cmd/layad \
 && cp "$ONNXRUNTIME_SHARED_LIBRARY_PATH" /out/libonnxruntime.so

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/layad /usr/local/bin/layad
COPY --from=build /out/libonnxruntime.so /usr/local/lib/libonnxruntime.so
ENV ONNXRUNTIME_SHARED_LIBRARY_PATH=/usr/local/lib/libonnxruntime.so \
    LAYA_MODELS_DIR=/models \
    LAYA_ADDR=:8080
EXPOSE 8080
ENTRYPOINT ["layad"]
