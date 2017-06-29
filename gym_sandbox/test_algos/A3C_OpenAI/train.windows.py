import argparse
import os
import subprocess
import sys
from six.moves import shlex_quote

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="./log", help='Log directory path')
    parser.add_argument('--env-id', default="police-killall-trigger-3dravel-v0", help='Environment id')

    args = parser.parse_args()
    num_workers = args.num_workers
    procAll = []
    pslog = open(os.path.join(args.log_dir, 'log.ps.out'), 'w')
    proc = subprocess.Popen(['python.exe',
                      'worker.py',
                      '--log-dir', args.log_dir,
                      '--env-id', '{}'.format(args.env_id),
                      '--num-workers', '{}'.format(num_workers),
                      '--job-name', 'ps'],
                     stdout=pslog,
                     stderr=pslog)
    procAll.append(proc)

    for i in range( num_workers):
        workerlog = open(os.path.join(args.log_dir, 'log.worker.{}.out'.format(i)), 'w')
        proc = subprocess.Popen(['python.exe',
                          'worker.py',
                          '--log-dir', args.log_dir,
                          '--env-id', '{}'.format(args.env_id),
                          '--num-workers', '{}'.format(num_workers),
                          '--job-name', 'worker',
                          '--task', '{}'.format(i)]
                         ,stdout=workerlog
                         ,stderr=workerlog
                         )
        procAll.append(proc)

    tblog = open(os.path.join(args.log_dir, 'log.tb.log'), 'w')
    proc = subprocess.Popen(['tensorboard.exe',
                      '--logdir', args.log_dir,
                      '--port', '12345'],
                     stdout=workerlog,
                     stderr=workerlog)
    procAll.append(proc)

    while True:
        cmd = input("exit all?")
        if cmd=="yes":
            for p in procAll:
                p.kill()
            break
