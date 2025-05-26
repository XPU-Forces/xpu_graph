#!/bin/bash

# commit message
commit_msg=`cat $1`
msg_re="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|release)(\(.+\))?: \w{1,100}"
error_msg="
[commit type](module): descriptions\n
1. commit type should be feat/fix/docs/style/refactor/perf/test/build/ci/chore/relase.\n
2. module should start with the second-level directory of repository if it belongs to, such as tools/....\n
3. descriptions should be less equal 100 words.\n
"

if [[ ! $commit_msg =~ $msg_re ]]
then
  echo -e $error_msg
  echo "Original Message is:"
  echo -e $commit_msg
  exit 1
fi
