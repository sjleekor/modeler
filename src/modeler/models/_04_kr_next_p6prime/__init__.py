"""P6′ — the next KR model (P6, ``_03_kr_next``) with one input mart replaced.

Preregistration draft: ``my/milestones/kr/modeling/plan/20261102_p6prime_prereg/
01_preregistration_draft.md`` (decision O-7: a new module that wraps the P6 code, a new view
folder). P6′ keeps P6's model, features, training procedure and judgement rules; only the
financial quarterly mart changes, from ``fin_quarterly_metric_vintage_dlr`` (P6) to
``fin_quarterly_metric_vintage_dlr_v4p``, which the new view folder carries under the P6 file name.

Nothing in ``_03_kr_next`` is modified. This package

* checks the P6′ tags (``kr-next-p6prime-frozen``, ``kr-next-p6prime-bundle-frozen``) instead of
  the P6 tags (:mod:`.tags`);
* builds the P6′ view folder from one raw snapshot and the P6 view folder (:mod:`.inputs`);
* wraps the P6 panel builder and training CLI (:mod:`.build_panel`, :mod:`.train`);
* holds the label-free input checks of the draft's §3 (:mod:`.checks`).
"""
