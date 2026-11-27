# Full-pipeline launch validation - 13 October 2026

The official prepared split counts are 5,062 train, 723 dev and 1,447 test claims.
All prepared-file hashes and cached dataset, PhoBERT and VnCoreNLP asset hashes
were verified against the existing pinned manifests. CUDA-enabled PyTorch
2.6.0+cu124 and the RTX 3050 Laptop GPU were available.

All 29 existing project tests passed during the preflight test run. Six new full
runner tests passed after correcting a Windows buffered-file lock issue. These
cover validation-only selection and ties, duplicate-run exclusion, completion
ledger resume/corruption rejection, cleanup boundaries, shared initialization,
three-seed sample SD and rejection of incomplete final reports. Python compilation,
CLI help, PowerShell launcher syntax and the repository diff check also passed.
Unrelated pytest plugins were disabled for the final runner test check.

The background process was launched at 10:33 Vietnam time. Check its project-local
status for current progress; this document does not claim training completion.

The real full-model resource check passed on eight long claims from the complete
training split. All encoder parameters remained trainable. Peak PyTorch allocated
memory was 2,947.4 MiB and reserved memory was 3,152.0 MiB. The first learning-rate
trial then completed its first eight-claim optimizer group in reranker epoch 1/3.

- Progress and logs: `runs/full-m0-m1-m2-20261013/` in the original project workspace.
- Checkpoints and detailed results are on C:. The physical path is recorded in
  `runs/full-m0-m1-m2-20261013/LOCATION.json`; this Windows app redirects AppData
  into its local application storage.
- The previous smoke/pilot outputs and checkpoints are preserved.
- The new runner's frozen source and manifest are saved in the new output folder.

This is all-data training with validation tuning and seeds 13, 42 and 2026. It has
no six-hour limit. Test evaluation is deferred until all selected settings and
training are finalized. No full-study scores are available at launch.

On 17 October 2026, the post-training M2 checkpoint checker was repaired after a
parameter-name mismatch interrupted the first lambda trial. Only `joint_checks`
in `vifactcheck/full.py` changed: probe names now map to the full nested
`PairEncoder.encoder` checkpoint names. Training code, model architecture, data,
hyperparameters, and completed weights were preserved.

All 39 tests passed, including new regressions for the nested names, rejection of
unchanged verifier/reranker encoders, and resuming a completed two-epoch M2 phase
without training or phase validation. The real saved M2 checkpoint also passed
the repaired GPU check: verification-only reranker-head gradient norm was
0.0034865267; all parameters in both encoders were trainable; first/last attention
query matrices changed in both encoders. This one-claim diagnostic used 1,344.9
MiB peak allocated CUDA memory and made no optimizer updates. That number is a
diagnostic measurement, not the memory requirement of full M2 training.

The original run plan and all 15 original source files are retained under
`recoveries/20261016-checkpoint-check/` in the physical checkpoint directory.
The original and corrected hashes, exact patch, test report, checkpoint hashes,
and real-model check are recorded there; `code_recovery.json` is also mirrored
beside the project status. The active source pins were updated for this single
post-training function repair, with the source/configuration guard retained.

The pipeline was relaunched at 00:01 Vietnam time on 17 October, reusing the
completed M2 epochs. Its current state is in the live status file; this recovery
record does not claim the full experiment is complete.
