# Statements quoted from the official dataset README

Source: `README.md` of `https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data`
at commit `5bc3c8d70cf8736d24b969972a3dd365017a9e24` (the same frozen commit as
`03_Code_files/reference/atlas_5bc3c8d7/`), cited in the manuscript as
`ATLASGitLab`.

Read on 10 October 2026 through the GitLab API
(`/api/v4/projects/atlas%2FATLAS-top-tagging-open-data/repository/files/README.md?ref=5bc3c8d7...`):

| Field | Value |
|---|---|
| size | 25510 bytes |
| git blob id | `423c543491b0821ec26b2896fa62467da360917a` |
| content SHA-256 | `4fe790030633f35a7b922eee619cb797813cddc0b77cbcd53f71e2e617873251` |
| last commit touching the file | `5bc3c8d70cf8736d24b969972a3dd365017a9e24` |

Each sentence below was checked to occur verbatim in that file (spelling,
including the original typos, unchanged). `verify_manuscript.py` checks that
every fragment the manuscript quotes is a substring of one of these lines.

Q1 (section "Jet Four-vector"):
**IMPORTANT**: The jet four-vector is not re-calculated from the systematic varied jet constituents in the datasets meant for asssessing experimental systematic uncertainties. This is because the jet four-vector is calibrated.

Q2 (section "Systematic uncertainties"):
A good estimate of the uncertainty requires that the background rejection is measured with sufficient precision that the difference in this metric between the systematic varied and nominal datasets is meaningful. In particular, the statistical uncertainty on the performance metrics must be smaller than this difference.

Q3 (section "Systematic uncertainties"):
For example it is recommended to use at least 5 million jets in evaluation when calculating the experimental uncertainties.

Q4 (section "Training with large data sets"):
The example training script solves this problem by only using 6 million of the jets in the training set. The user should tune how many jets are taken from the training set, selecting as many jets as will fit within memory constraints. In practice obtaining state-of-the-art performance will require training on the entire set, which will require data piping.

Q5 (systematics table):
Vary the energy scale of jet constituents reconstructed with the calorimeter

Q6 (section "High Level Quantities"): the section lists the fifteen `fjet_*`
branches and says only that they "summarize" the constituent information; it
contains no statement on whether they are recomputed in the systematic samples.
