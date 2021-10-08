from pssh.clients import ParallelSSHClient
from gevent import joinall
import json
import os
import time
#hosts = ['host1', 'host2', 'host3', 'host4']
tot_host=[]
tot=[]
start_time = time.time()
tot_start=[start_time]
#hosts = [,'192.168.137.137','192.168.137.179','192.168.137.178']
hosts = ['192.168.137.179']
client = ParallelSSHClient(hosts, user='pi', password='raspberri')
#cmds = client.scp_send('../dnn minibatch script/TrainData1.json','/home/pi/py_scripts/TrainData.json', recurse=True)
# sending training data from seed node to service node.
copy_args = [{'local_file': '../dnn minibatch script/TrainData1.json',
             'remote_file': '/home/pi/py_scripts/TrainData.json',},
             {'local_file': '../dnn minibatch script/TrainData2.json',
             'remote_file': '/home/pi/py_scripts/TrainData.json',},
             {'local_file': '../dnn minibatch script/TrainData3.json',
             'remote_file': '/home/pi/py_scripts/TrainData.json',},
             {'local_file': '../dnn minibatch script/TrainData4.json',
             'remote_file': '/home/pi/py_scripts/TrainData.json',}]
# sending initial parametes t from seed node to service node.
      
comd = client.copy_file('%(local_file)s', '%(remote_file)s',copy_args=copy_args)
copy_args = [{'local_file': '../dnn minibatch script/parameters1.json',
             'remote_file': '/home/pi/py_scripts/parameters1.json',},
             {'local_file': '../dnn minibatch script/parameters1.json',
             'remote_file': '/home/pi/py_scripts/parameters1.json',},
             {'local_file': '../dnn minibatch script/parameters1.json',
             'remote_file': '/home/pi/py_scripts/parameters1.json',},
             {'local_file': '../dnn minibatch script/parameters1.json',
             'remote_file': '/home/pi/py_scripts/parameters1.json',}]
comds = client.copy_file('%(local_file)s', '%(remote_file)s',copy_args=copy_args)
cmd='python3 /home/pi/py_scripts/decode_ccrip.py'
#shells = client.open_shell()
#client.run_shell_commands(shells,cmd)
joinall (comd, raise_error=True)
outputs = client.run_command(cmd)
for host in outputs:
    for line in host.stdout:
        print(line)
    command=client.scp_recv('/home/pi/py_scripts/TestData.json', '../dnn minibatch script/TestData.json', recurse=True)
    end_time = time.time()
    tot_host.append(end_time)
#client.copy_file('/home/pi/py_scripts/TestData.json', '../dnn minibatch script/Test.json', recurse=False, copy_args=None)
#command=client.scp_recv('/home/pi/py_scripts/TestData.json', '../dnn minibatch script/TestData.json', recurse=True)
total = end_time-start_time
print(tot_start)
tot=[x1-x2 for x1,x2 in zip(tot_host,tot_start)]
joinall (command, raise_error=True)
print(f'Total time is {total}')
print(tot_host)
print(tot)
#client.join_shells(shells)






