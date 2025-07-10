#!/bin/bash
current_dir=`pwd`
script_dir=$(cd $(dirname $0); pwd)
BYTED_CLOUD_AICOMPILER_SECRET=$1

cd ${current_dir}/dist
TAR_X86_64=`ls *.whl`
WHL_X86_64=`ls *.tar*`
echo release: $TAR_X86_64, $WHL_X86_64

# 1. 获取 JWT_TOKEN
JWT_TOKEN=`curl -s -I -X GET  https://cloud.bytedance.net/auth/api/v1/jwt -H "Authorization: Bearer ${BYTED_CLOUD_AICOMPILER_SECRET}" | awk -F': ' '/^x-jwt-token/ {print $2}'`

if [ -n $JWT_TOKEN ]; then
    echo JWT_TOKEN get success.
else
    echo JWT_TOKEN get failed.
    exit 1
fi

# 2. 上传版本到 byted pypi luban
curl --location 'https://bytedpypi.byted.org/' \
--header "X-Jwt-Token: ${JWT_TOKEN}" \
--form "pypi.package=${PKG}" \
--form "asset0.filename=${TAR_X86_64}" \
--form "asset1.filename=${WHL_X86_64}" \
--form "asset0=@${TAR_X86_64}" \
--form "asset1=@${WHL_X86_64}"

cd ${current_dir}
