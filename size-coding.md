# Size Coding Challenge

The goal of this challenge is to optimize the 8k intro, **"Night Ride"**, and shrink it down as much as possible, while maintaining its visual quality and performance. The source code is available in this repository, and we invite participants to contribute by making the code smaller while preserving the demo’s overall experience.

The original release was around 8kB, but we expect that there are many ways to reduce its size.

This challenge is a collaborative effort to push the boundaries of size optimization. The goal is to get a better understanding of what's possible in 8kB and what techniques can be used. The findings may be reused in future demoscene productions, and we plan to write articles about the best techniques. Collaborations and discussions (e.g. on the bug tracker) are encouraged.

## Challenge Rules

1. **Source Code and Tooling**:
    * Participants are allowed to modify the **source code** (before it is minified via Shader Minifier).
    * **Modifications to the generated (minified) code are not allowed**, as they would affect maintainability.
    * Code should be **commented and maintainable**, ensuring that others can follow and iterate on changes.
    * **PRs that improve Shader Minifier** itself are welcome and will be reviewed separately.
    * We do not plan to change the **tooling** (e.g., changing the compiler), but modifying **compiler options** is permitted for optimization.
    * **Changes that significantly reduce performance** (e.g., reducing the framerate) will be **rejected**.
2. **Visual Quality**:
    * The goal is to keep the visuals **as close as possible** to the original version.
    * Contributions that **don’t alter the visuals** will be classified as **pure improvements**.
    * Contributions with **minor visual changes** that don’t significantly impact the experience will be classified as **impure improvements** and reviewed by a human for quality control.
    * Other changes that affect the visuals can be discussed in the bug tracker, and may be considered if the visual quality is considered "as good".
    * We’ll provide a **test infrastructure** (screenshot comparison) to help verify if a change qualifies as pure.
3. **Minimum Contribution**:
    * To limit the number of submissions, each PR must save **at least 5 bytes**.
    * The demo will be compressed using Crinkler with the `Snapshot` configuration to ensure quick iteration and fast compression times, allowing participants to focus on coding rather than lengthy compression processes.
4. **Other Improvements**:
    * Changes that **improve the codebase** but don’t affect the size (e.g., refactoring, improving readability) are accepted but won’t count toward the leaderboards.
    * PRs that improve the **Editor mode** to enhance quality of life during development are welcome. However, these will not count toward the challenge and will not appear on the leaderboard.
5. **Submission Process**:
    * All collaboration will happen via **pull requests** to the repository, on the branch `size`.
    * Once a PR is merged, the leaderboard will be updated to reflect the number of bytes saved.

## Leaderboards

1. **Pure Improvements**: This leaderboard tracks size reductions with **no visual changes**.
2. **Impure Improvements**: This leaderboard tracks size reductions that involve **minor acceptable visual changes**.

Both leaderboards will display the **number of bytes removed** by each participant.
