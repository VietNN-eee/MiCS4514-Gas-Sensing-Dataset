# -*- coding: utf-8 -*-
"""Run the full re-analysis and regenerate all figures.

Usage:  python run_all.py            (all analyses, then figures; about 1-2 h on a laptop)
        python run_all.py figures    (figures only, from the result files already in ../results)
Data:   the four Excel files in the parent folder of this code folder (or set ENOSE_DATA_DIR).
"""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ('run_cls.py', ['table2', 'pairs', 'boot', 'alt', 'svm', 'robust']),   # Table II, Tables S6, S8-S12; confusion matrices for Fig. 5(a), S6
    ('run_fig7b.py', [None]),                                               # Fig. 5(b), graphical abstract
    ('run_r2.py', [None]),                                                  # Table S7: training-only vs transductive statistics, same folds
    ('run_reg.py', ['table5', 'samemre', 'plsk', 'loco', 'loco2']),        # Table IV, Fig. 7, Tables S16-S18
    ('run_samermse.py', [None]),                                            # same-sensor RMSE (diagonal of Table S11)
    ('run_r2c.py', ['tab5', 'two', 'loco']),                               # nested PLS (Table IV, Tables S17-S18)
    ('run_misc.py', ['ablation', 'refdesign', 'dscoral', 'lod']),          # Fig. 6, Table III, Tables S4-S5, S13-S15
    ('run_r2b.py', [None]),                                                 # within-segment noise, lag-one autocorrelation (Table S5)
]
def run(script, part):
    cmd = [sys.executable, os.path.join(HERE, script)] + ([part] if part else [])
    t0 = time.time(); print('>>', ' '.join(cmd[1:]), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
    print('   done in %.0f s' % (time.time() - t0), flush=True)
if __name__ == '__main__':
    if not (len(sys.argv) > 1 and sys.argv[1] == 'figures'):
        for script, parts in STEPS:
            for p in parts: run(script, p)
    run('make_figures.py', None)
