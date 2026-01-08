module.exports = {
  npmPublish: false,
  plugins: [
    ["@semantic-release/commit-analyzer", {
      releaseRules: [
        { type: "major", release: "major" },
        { type: "release", release: "major" },
      ],
      parserOpts: {
        noteKeywords: ["BREAKING CHANGE", "BREAKING CHANGES", "BREAKING"]
      }
    }],
    ["@semantic-release/exec", {
      prepareCmd: `
        set -e
        VER=\${nextRelease.version}
        ##vso[build.updatebuildnumber]\${nextRelease.version}

        docker login -u "$DOCKER_HUB_USER" -p "$DOCKER_HUB_PASSWORD"

        docker build -f Dockerfile \
          -t "$DOCKER_HUB_USER/koleg.ai:$VER" \
          "$LOCAL_PATH"

        docker push "$DOCKER_HUB_USER/koleg.ai:$VER"
      `,
      successCmd: `
        set -e
        ##vso[build.addbuildtag]release
        ##vso[build.addbuildtag]\${nextRelease.type}
        ##vso[build.addbuildtag]\${nextRelease.version}
      `,
    }],
    "@semantic-release/git"
  ],
  branches: ['master', 'preview'],
  publish: []
}
