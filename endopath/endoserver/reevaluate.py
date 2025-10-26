"""Utility to re-run inference for cases using strict models.

Usage examples:
  python reevaluate.py --slide-id SLIDE-49E1FC56
  python reevaluate.py --scope failed
  python reevaluate.py --scope all --model program3
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import sys
from typing import Optional

from app.db import get_conn
from app.inference_strict import run_inference


def reevaluate_slide(slide_id: str, model: Optional[str] = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, status, prediction FROM cases WHERE slide_id = ?", (slide_id,))
    row = cur.fetchone()
    if not row:
        print(f"Slide ID not found: {slide_id}")
        return 1
    case_id = row[0]
    print(f"Re-evaluating case {case_id} (slide {slide_id}) with model={model or 'case-default'}...")
    run_inference(case_id, model_override=model)
    cur.execute("SELECT status, prediction, confidence FROM cases WHERE id = ?", (case_id,))
    s, p, c = cur.fetchone()
    print(f"Result: status={s}, prediction={p}, confidence={c}")
    return 0


def reevaluate_scope(scope: str, model: Optional[str] = None) -> int:
    where = ''
    if scope == 'failed':
        where = "WHERE status='failed'"
    elif scope == 'completed':
        where = "WHERE status='completed'"
    elif scope == 'pending':
        where = "WHERE status='pending'"

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT id, slide_id FROM cases {where}")
    rows = cur.fetchall()
    if not rows:
        print("No matching cases")
        return 0
    print(f"Re-evaluating {len(rows)} cases (scope={scope}) with model={model or 'case-default'}...")
    for cid, slide in rows:
        print(f" - {slide} ({cid})")
        run_inference(cid, model_override=model)
    print("Done.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--slide-id', help='Specific slide ID to reevaluate')
    g.add_argument('--scope', choices=['all', 'failed', 'completed', 'pending'], help='Batch reevaluation scope')
    parser.add_argument('--model', choices=['program3', 'program1'], help='Override model key')
    args = parser.parse_args(argv)

    if args.slide_id:
        return reevaluate_slide(args.slide_id, args.model)
    else:
        return reevaluate_scope(args.scope, args.model)


if __name__ == '__main__':
    sys.exit(main())
