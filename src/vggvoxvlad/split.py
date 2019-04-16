
import librosa
import pandas as pd
import src.vggvoxvlad.utils_dan as ut_d
import numpy as np
import src.vggvoxvlad.model as model
import matplotlib.pyplot as plt


def make_network(weight_path, args, input_dim=(257, None, 1), num_class=1251):
	network_eval = model.vggvox_resnet2d_icassp(input_dim=input_dim,
                                                num_class=num_class,
                                                mode='eval', args=args)
	network_eval.load_weights(weight_path, by_name=True)

	return network_eval


def voxceleb1_split(path, network, split_seconds=3, n=3, 
                    win_length=400, sr=16000, hop_length=160, 
                    n_fft=512, spec_len=250, n_classes=1251):

	vc_df = pd.read_csv('../data/raw/vox1_meta.txt',sep = '\t',skiprows=0)
	vc_df['class'] = pd.to_numeric(vc_df['VoxCeleb1 ID'].str.replace('id','')) - 10001

	x, sr = librosa.load(path, sr)

	df_audio = pd.DataFrame({"Time":np.arange(x.shape[0])/sr,"Amplitude":x})

	num_segments = int(np.floor(df_audio['Time'].max()/split_seconds))

	result_list = []
	t = 0
	dt = split_seconds  # how often to split audio file for predicting
	
	for i in range(num_segments):
	    
	    df_out = pd.DataFrame([])
	    
	    df_tmp = df_audio.copy()
	    df_tmp = df_tmp[(df_tmp['Time'] >= t) & (df_tmp['Time'] < (t+dt))]
	    
	    amp = np.array(df_tmp['Amplitude'])
	    
	    specs = ut_d.load_data(amp, win_length=win_length, sr=sr,
	                         hop_length=hop_length, n_fft=n_fft,
	                         spec_len=spec_len, mode='eval')
	    specs = np.expand_dims(np.expand_dims(specs, 0), -1)

	    v = network.predict(specs)
	    
	    # find top 3
	    v = v.reshape(n_classes)
	    ind = v.argsort()[-n:][::-1]
	    
	    for i in range(n):
	        df_out = df_out.append(pd.DataFrame({'Time (s)': t+dt,
	                                             'Speaker': vc_df['VGGFace1 ID'][ind[i]].replace('_', ' '),
	                                             'Probability': v[ind][i], 
	                                             'Country': vc_df['Nationality'][ind[i]],
	                                             'Gender': vc_df['Gender'][ind[i]]}, index=[0]), ignore_index=True)
	    #df_out = df_out.set_index('Time (s)')
	    result_list.append(df_out)
	    print(df_out)
	    
	    t += dt

	return result_list


def plot_split(result_list, num_speakers = 2):
    t = []
    prob = []
    name = []
    
    for df in result_list:
        if df['Probability'][0] >= 0.5:
            t.append(df['Time (s)'][0])
            prob.append(df['Probability'][0])
            name.append(df['Speaker'][0])
    t = np.array(t)
    prob = np.array(prob)
    
    from collections import Counter
    speaker_dict = Counter(name)
    sorted_speaker = sorted(speaker_dict.items(), key=lambda kv: kv[1])
    sorted_speaker = sorted_speaker[-num_speakers:]
    names = [i[0] for i in sorted_speaker]
    
    for i in range(len(name)):
        if name[i] not in names:
            name[i] = 'Other'
            
    import seaborn as sns
    plt.figure(figsize=(16, 2))
    by_school = sns.barplot(x=t, y=prob, hue=name);
    plt.xlabel('Time (s)');
    plt.ylabel('Probability');
    
    for item in by_school.get_xticklabels():
        item.set_rotation(90)










