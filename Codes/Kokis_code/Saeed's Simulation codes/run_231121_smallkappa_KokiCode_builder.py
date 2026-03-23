#! /usr/bin/env python

import subprocess
import numpy
import os

partition_info=['normal',16]; time_str='2-00:00:00' # = [partition,ncores]
#partition_info=['debug',16]; time_str='2:00:00' # = [partition,ncores]
#partition_info=['Cosmo',16]; time_str='4-00:00:00' # = [partition,ncores]
#partition_info=['clint',48]; time_str='2-00:00:00' # = [partition,ncores]
#partition_info=['128s',16]; time_str='4-00:00:00' # = [partition,ncores]
#partition_info=['CMT',32]; time_str='6-00:00:00' # = [partition,ncores]
#partition_info=['smallmem',8]; time_str='4-00:00:00' # = [partition,ncores] 
project_name=os.getcwd().split('/')[-3]
myemail=os.environ["MYEMAIL"]
charge_str="inverted_mbl"

#Log the current submission
logstr='''run_231121_smallkappa_KokiCode:
For L=18: if l == 18 and G<0.3 and vnum%4==0 and Ee<0:



***write me***

Codebase: 
Cluster: ls5
gittag: '''

#Do the git-ing
cmd='git commit -a -m "Commit before run run_231121_smallkappa_KokiCode" > /dev/null'
print(cmd)
subprocess.call(cmd,shell=True)

cmd='gittag.py > temp.temp'
subprocess.call(cmd,shell=True)
logstr+=open('temp.temp','r').readline()
subprocess.call('rm temp.temp',shell=True)

open('run_231121_smallkappa_KokiCode.log','w').write(logstr)

#Setup the versionmap and qsub files
vmap_file=open('versionmap.dat','w')
vmap_file.write('vnum\tL\n')

task_file=open('run_231121_smallkappa_KokiCode.task','w')
template_file='run_231121_smallkappa_KokiCode.template'
template_contents=open(template_file,'r').read()



L = [5,6,7,8,9,10]
J=1.0
F = numpy.round(numpy.arange(0.0, 0.5, 0.04),3)
phi_list = [0,0.25, 0.5,0.75, 1]
Nt=35
T = 1
Gamma_list = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06,] + list(numpy.round(numpy.arange(0.10, .53, 0.06),3))+list(numpy.round(numpy.arange(0.6, 2.81, 0.15),3))
Rmaxlist= [1,1,1,1,1,1]
vnum=0
vnumt=0
#random.seed(1)
Threads = 1

for h in [0, 0.1, 1.0, 3.0]:
        for L, Rmax in zip([4, 5, 6, 7, 8, 9, 10, 11], [12, 12, 12, 12, 12, 12, 12, 12] ):
                for Gamma in Gamma_list:
                        for R in numpy.round(range(1, Rmax+1),4):
                                if L==4 and R%6==0 and h>0 and Gamma<1.1 :
                                        #if phi in [2.0,3.0,4.0]:
                                        #if f in [0.2, 0.4, 1.1] and 2.1<phi<4.0:
                                        qsub_file=template_file.replace('.template','_'+str(vnum)+'.qsub')
                                        fout=open(qsub_file,'w')
                                        contents=template_contents.replace('###',str(vnum))
                                        contents=contents.replace('*project*',project_name)
                                        # *Omega_d*  *kappa_R*  *NR*  *US*  *tmax*  *dt*
                                        contents=contents.replace('*hhh*',str(h))
                                        contents=contents.replace('*Gamma*',str(Gamma))
                                        contents=contents.replace('*LLL*',str(L))
                                        contents=contents.replace('*ItNum*',str(300))
                                        contents=contents.replace('*runseed*',str(vnum*400))
                                        vmap_file.write(str(vnum)+'\t'+str(L)+'\t'+str(Gamma)+'\t'+str(R)+'\n')
                                        print('vnum: ', vnum,', L = ', L,', Gamma = ', Gamma,', R: ',R,  )
                                        print('bash run_231121_smallkappa_KokiCode_'+str(vnum)+'.qsub')
                                        #print(numpy.arange(-0.45*l,(d-1)*Omega*0.5,((d-1)*Omega*0.5-(-0.45*l))/12))
                                        task_file.write('bash run_231121_smallkappa_KokiCode_'+str(vnum)+'.qsub\n')
                                        vnumt+=1
                                        fout.write(contents)
                                        fout.close()
                                vnum+=1










n_cpu_prtsk = Threads
n_cores=(vnumt+0)
#n_nodes=int(numpy.ceil(float(vnum)/partition_info[1]))
n_nodes=int(numpy.ceil(float(n_cpu_prtsk*vnumt)/partition_info[1]))
print(n_cpu_prtsk, vnumt, n_nodes)


#n_nodes=int(numpy.ceil(float(vnumt)/partition_info[1]))
# Pad to an even multiple of cores per node
#n_cores=n_nodes*partition_info[1]
for j in range(vnumt,n_cores):
        task_file.write('echo "Fake run"\n')

# Finally output sbatch file
contents=open('run_231121_smallkappa_KokiCode.sbatch.template','r').read()
contents=contents.replace('*JobDetails*',str('L==4 and R%6==0 and h>0 and Gamma<1.2'))
contents=contents.replace('*nnn*',str(n_cores)) # The total number of processors
contents=contents.replace('*NNN*',str(n_nodes)) # The total number of nodes
contents=contents.replace('*ttt*',time_str) # The wall clock time per processor
contents=contents.replace('*ccc*',str(n_cpu_prtsk)) # Number of cpus per task
contents=contents.replace('*partition*',partition_info[0]) # Partition
contents=contents.replace('*myemail*',myemail) # Email
contents=contents.replace('*charge*',charge_str) # Project to charge to
open('run_231121_smallkappa_KokiCode.sbatch','w').write(contents)





