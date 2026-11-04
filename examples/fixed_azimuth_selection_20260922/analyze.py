#!/usr/bin/env python3
"""Select a repeatable flapwise-response azimuth from the 11.25m/s idling record.

This screens a fixed-angle candidate; a rotating response includes motion history
and cannot establish the worst stationary-rotor equilibrium. No load is retuned.
"""
from pathlib import Path
import argparse,hashlib,json,csv
import numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'paper_drafts/NAWEA_QBladeBridge/presentation/data/full_load_histories/moments_full_600s_U11p25_samples.npz'
MODELS=[('qblade','QBlade','#1B1B1B'),('spectral','Spectral VorLap','#348ABD'),('hybrid','Hybrid VorLap','#A60628')]
def binned(signal,angle,rev,width):
 out=np.empty((10,360))
 for center in range(360):
  mask=np.abs((angle-center+180)%360-180)<width/2
  for n in range(10):
   values=signal[mask&(rev==n)];assert len(values)>0
   out[n,center]=np.sqrt(np.mean(values**2))
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=DEFAULT);ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'evidence');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 with np.load(args.data) as d:
  t=d['physical_time_s'];az=d['azimuth_unwrapped_deg'];dt=float(np.median(np.diff(t)));assert len(t)==80000 and np.isclose(dt,.0075,atol=1e-12)
  phase=float(az[0]-6*t[0]);assert np.max(np.abs(az-(phase+6*t)))<1e-9
  rev=np.floor((np.arange(len(t))*.0075)/60).astype(int);assert np.array_equal(np.bincount(rev),np.full(10,8000));angle=az%360
  filtered={};metrics={}
  for key,_,_ in MODELS:
   y=d[key+'_y_Nm'];assert np.isfinite(y).all()
   filtered[key]=sosfiltfilt(butter(4,.5,fs=1/dt,btype='highpass',output='sos'),y)
   metrics[key]=binned(filtered[key],angle,rev,10)
  med=np.median(metrics['hybrid'],axis=0);selected=int(np.argmax(med));sensitivity=[]
  for cutoff in [.2,.5,1.]:
   yy=sosfiltfilt(butter(4,cutoff,fs=1/dt,btype='highpass',output='sos'),d['hybrid_y_Nm'])
   for width in [5,10,20]:
    v=binned(yy,angle,rev,width);values=np.median(v,axis=0);n=int(np.argmax(values));sensitivity.append({'cutoff_hz':cutoff,'window_width_deg':width,'maximizing_angle_deg':n,'peak_median_rms_Nm':float(values[n])})
  # An independent phase-window standard-deviation metric (no filter) checks
  # that the lobe is not created by the chosen high-pass filter.
  detrended=np.empty((10,360))
  for center in range(360):
   mask=np.abs((angle-center+180)%360-180)<5
   for n in range(10):
    idx=np.flatnonzero(mask&(rev==n));# unwrap local time across the circularwindow if needed
    local_angle=(angle[idx]-center+180)%360-180
    coeff=np.polyfit(local_angle,d['hybrid_y_Nm'][idx],1)
    yy=d['hybrid_y_Nm'][idx]-np.polyval(coeff,local_angle);detrended[n,center]=np.sqrt(np.mean(yy**2))
  ldpeak=int(np.argmax(np.median(detrended,axis=0)))
  rows=[]
  for center in range(360):
   row={'azimuth_deg':center}
   for key,_,_ in MODELS:
    v=metrics[key][:,center];row.update({key+'_median_rms_Nm':float(np.median(v)),key+'_q25_rms_Nm':float(np.quantile(v,.25)),key+'_q75_rms_Nm':float(np.quantile(v,.75))})
   rows.append(row)
  with (args.output/'azimuth_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  with (args.output/'hybrid_revolution_rms.csv').open('w') as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(['azimuth_deg']+[f'revolution_{i+1}_rms_Nm' for i in range(10)]);w.writerows([[i,*metrics['hybrid'][:,i]] for i in range(360)])
  receipt={'source_data':str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),'source_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wind_speed_mps':11.25,'seed':4433456,'station':'Blade 1, z/H = 0.23','channel':'Local Y flapwise moment [Nm]','time_window_s':[60,660],'samples':len(t),'time_step_s':.0075,'initial_recorded_phase_deg':phase,'angular_speed_deg_per_s':6.,'criterion':'Maximize median across 10 revolutions of high-pass flapwise RMS in a 10-degree circular window centered at each 1-degree grid angle. Zero-phase fourth-order Butterworth cutoff 0.5 Hz isolates fast oscillation from slow 1 rpm load variation.','selected_azimuth_deg':selected,'selected_statistics':rows[selected],'comparison_30deg':rows[30],'hybrid_each_revolution_selected_rms_Nm':metrics['hybrid'][:,selected].tolist(),'hybrid_each_revolution_30deg_rms_Nm':metrics['hybrid'][:,30].tolist(),'sensitivity':sensitivity,'independent_local_linear_detrend_peak_deg':ldpeak,'scope':'Candidate selected from one matched-seed rotating record, not a proven worst fixed angle. A fixed-rotor run changes wake, feedback and response history; no equilibrium or universal worst angle is claimed.'}
  (args.output/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
  plt.rcParams.update({'font.family':'Open Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
  fig,axs=plt.subplots(1,2,figsize=(12.3,4.2),gridspec_kw={'width_ratios':[1,1.1]})
  for key,label,color in MODELS:
   v=metrics[key];axs[0].plot(range(360),np.median(v,axis=0),color=color,label=label)
  axs[0].fill_between(range(360),np.quantile(metrics['hybrid'],.25,axis=0),np.quantile(metrics['hybrid'],.75,axis=0),color='#A60628',alpha=.12,label='Hybrid middle 50%')
  for ax in axs:ax.axvline(selected,color='#A60628',ls='--',lw=1);ax.axvline(30,color='#888888',ls=':',lw=1);ax.set(xlim=(0,360),xticks=np.arange(0,361,60),xlabel='QBlade rotor azimuth [deg]')
  axs[0].set(ylabel='Flapwise high-pass RMS [N m]',title=f'(a) Repeated response peaks near {selected}°');axs[0].legend(frameon=False,fontsize=9);axs[0].grid(alpha=.18)
  im=axs[1].imshow(metrics['hybrid'],origin='lower',aspect='auto',extent=(-.5,359.5,.5,10.5),cmap='magma');axs[1].set(ylabel='Retained revolution',title='(b) Hybrid response across 10 revolutions');fig.colorbar(im,ax=axs[1],label='Flapwise RMS [N m]',pad=.02)
  fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(args.output/f'azimuth_selection.{ext}',dpi=230)
  print(json.dumps({k:receipt[k] for k in ['selected_azimuth_deg','selected_statistics','comparison_30deg','sensitivity','independent_local_linear_detrend_peak_deg']},indent=2))
if __name__=='__main__':main()
