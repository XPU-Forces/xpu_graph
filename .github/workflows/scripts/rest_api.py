import argparse
import json
import os
import sys
from functools import cache, partialmethod
from typing import AnyStr, List

import urllib3

__XPU_GRAPH_DIR__ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "xpu_graph"
)
sys.path.append(__XPU_GRAPH_DIR__)

from version import __version__

__ACTIONS__ = {"tag": "git/tags", "release": "releases", "generate_release_note": "releases/generate-notes"}

parser = argparse.ArgumentParser()
parser.add_argument("--private_token", type=str, required=True, help="")
parser.add_argument("--artifacts", type=str, required=True, nargs="+", dest="artifacts", help="")
parser.add_argument("--branch_or_commit", "-boc", type=str, dest="branch_or_commit", default="master", help="")
parser.add_argument(
    "--tag_name",
    type=str,
    default=__version__,
    help="",
)
parser.add_argument("--repo", type=str, default="BD-Seed-HHW/xpu_graph", help="")
parser.add_argument("--actions", type=str, choices=__ACTIONS__.keys(), default="release", help="")
parser.add_argument(
    "--description", "-desc", type=str, dest="desc", nargs="?", default=f"Release xpu-graph {__version__}"
)
parser.add_argument(
    "--tagger",
    type=str,
    help="The input format should be username+email",
    default="dummy+dummynode@github.com",
)


class AutoAttr:
    @classmethod
    def set_attr(cls, attr_name, request_type):
        def decorator(func):
            setattr(cls, attr_name, partialmethod(func, request_type=request_type))
            return func

        return decorator


class GitHubRESTApi(AutoAttr):
    __URL_FMT__ = "https://api.github.com/repos/{repo}/{action}"

    def __init__(self, repo, private_token, user_name="", email=""):
        self.repo = repo
        self.private_token = private_token
        self.user_name = user_name
        self.email = email

        self.http = urllib3.PoolManager()
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.private_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    @AutoAttr.set_attr("delete", "DELETE")
    @AutoAttr.set_attr("get", "GET")
    @AutoAttr.set_attr("post", "POST")
    def __call__(self, url, headers, body, request_type="POST", status_ok=201):
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=True).encode("utf-8")
        resp = self.http.request(request_type, url, body=body, headers=headers)
        if status_ok:
            if resp.status == status_ok:
                return resp
            else:
                raise RuntimeError(
                    "Http status: {}, data:{}\n Original request to {} is: headers={} body={}".format(
                        resp.status, json.loads(resp.data.decode("utf-8")), url, headers, body
                    )
                )
        else:
            return resp

    def create_tag(self, tag_name, commit_id, desc):
        @cache
        def get_url():
            return self.__URL_FMT__.format(repo=self.repo, action=__ACTIONS__["tag"])

        url = get_url()
        data = {
            "tag": tag_name,
            "message": desc,
            "object": commit_id,
            "type": "commit",
            "tagger": {
                "name": self.user_name,
                "email": self.email,
            },
        }
        return self.post(url, self.headers, data)

    def create_release(
        self, tag_name, commit_id, desc, draft=False, prerelease=False, generate_release_notes=False, make_latest=False
    ):
        @cache
        def get_url():
            return self.__URL_FMT__.format(repo=self.repo, action=__ACTIONS__["release"])

        url = get_url()
        data = {
            "tag_name": tag_name,
            "target_commitish": commit_id,
            "name": tag_name,
            "body": desc,
            "draft": draft,
            "prerelease": prerelease,
            "generate_release_notes": generate_release_notes,
            # "make_latest": make_latest,
        }
        return self.post(url, self.headers, data)

    def query_release(self, tag_name):
        @cache
        def get_url(tag_name):
            return self.__URL_FMT__.format(repo=self.repo, action=f"releases/tags/{tag_name}")

        url = get_url(tag_name)
        return self.get(url, self.headers, None, status_ok=None)

    def delete_release(self, release_id):
        @cache
        def get_url(release_id):
            return self.__URL_FMT__.format(repo=self.repo, action=f"releases/{release_id}")

        url = get_url(release_id)
        return self.delete(url, self.headers, None, status_ok=204)

    def upload_asset(self, url, file_path):
        headers = self.headers.copy()
        headers["Content-Type"] = "application/octet-stream"
        with open(file_path, "rb") as f:
            data = f.read()
        return self.post(url, headers, data, status_ok=201)

    def release_artifact(
        self,
        tag_name,
        commit_id,
        desc,
        artifacts: List[AnyStr],
        draft=False,
        prerelease=False,
        generate_release_notes=False,
        make_latest=False,
    ):
        resp = self.query_release(tag_name)
        if resp.status == 200:
            self.delete_release(json.loads(resp.data.decode("utf-8"))["id"])

        resp = self.create_release(tag_name, commit_id, desc, draft, prerelease, generate_release_notes, make_latest)
        json_data = json.loads(resp.data.decode("utf-8"))

        base_url = json_data["upload_url"].replace("{?name,label}", "?name={filename}")
        for artifact in artifacts:
            filename = os.path.basename(artifact)
            upload_url = base_url.format(filename=filename)
            self.upload_asset(upload_url, artifact)


if __name__ == "__main__":
    args = parser.parse_args()
    client = GitHubRESTApi(args.repo, args.private_token, *args.tagger.split("+"))
    client.release_artifact(args.tag_name, args.branch_or_commit, args.desc, list(args.artifacts))
