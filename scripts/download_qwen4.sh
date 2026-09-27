#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${project_root}/models"
model_path="${model_dir}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
partial_path="${model_path}.part"
model_url="https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
model_sha256="3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"

mkdir -p "${model_dir}"
if [[ -f "${model_path}" ]]; then
  if echo "${model_sha256}  ${model_path}" | sha256sum --check --status; then
    printf 'Model đã tồn tại và đúng SHA-256: %s\n' "${model_path}"
    exit 0
  fi
  printf 'Model đã tồn tại nhưng sai SHA-256: %s\n' "${model_path}" >&2
  printf 'Hãy đổi tên hoặc xóa file sai trước khi tải lại.\n' >&2
  exit 1
fi

curl -L --fail --continue-at - --output "${partial_path}" "${model_url}"
echo "${model_sha256}  ${partial_path}" | sha256sum --check
mv "${partial_path}" "${model_path}"
printf 'Đã tải và xác minh: %s\n' "${model_path}"
