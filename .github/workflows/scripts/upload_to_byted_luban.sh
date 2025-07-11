#!/bin/bash
set -e
current_dir=`pwd`
script_dir=$(cd $(dirname $0); pwd)
BYTED_CLOUD_AICOMPILER_SECRET=$1
OAuth2_Client_secret=$2
PKG=byted-xpu-graph
cd ${current_dir}/dist
TAR_X86_64=`ls *.whl`
WHL_X86_64=`ls *.tar*`
echo release: $TAR_X86_64, $WHL_X86_64

# 1. 通过服务号获取个人 JWT_TOKEN
# 1.1 获取平台 token
TOKEN=$(curl -X GET 'https://cloud.bytedance.net/auth/api/v1/jwt' -H "Authorization: Bearer ${BYTED_CLOUD_AICOMPILER_SECRET}" -s -D - -o /dev/null | grep -i 'x-jwt-token' | awk '{print $2}' | tr -d '\r')
if [ -n $TOKEN ]; then
    echo get aicompiler token success.
else
    echo get aicompiler token failed.
    exit 1
fi

# 1.2 用平台 token 换取个人 token
JWT_TOKEN=$(curl -s --location --request POST 'https://cloud.bytedance.net/auth/api/v1/token' \
--header "X-Jwt-Token: $TOKEN" \
--header 'Content-Type: application/json' \
        --data-raw "{
         "client_id": "cli_XgwbKu66",
         "client_secret": "$OAuth2_Client_secret",
         "redirect_uri": "",
         "grant_type": "authorization_code",
         "auth_type": "custom",
         "username": "wangmingfa",
         "lark_auth_req": true
 }" | jq -r '.access_token')
if [ -n $JWT_TOKEN ]; then
    echo get wangmingfa token success.
else
    echo get wangmingfa token failed.
    exit 1
fi

# 2. 上传版本到 byted pypi luban
curl --http1.1 --location 'https://bytedpypi.byted.org/' \
--header "X-Jwt-Token: ${JWT_TOKEN}" \
--form "pypi.package=${PKG}" \
--form "asset0.filename=${TAR_X86_64}" \
--form "asset1.filename=${WHL_X86_64}" \
--form "asset0=@${TAR_X86_64}" \
--form "asset1=@${WHL_X86_64}"

cd ${current_dir}
