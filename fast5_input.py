from random import shuffle
from glob import glob
from tqdm import tqdm
import numpy as np
import h5py


def gen(l, inf=True):
    while True:
        for ll in l:
            yield ll
        if not inf:
            break
            
def fast5_to_valnlab(file_path, segment_length, training=True):
    with h5py.File(file_path,'r') as input_data:
        for read_name in input_data['Raw/Reads']:
            raw_signals = input_data['Raw/Reads'][read_name]['Signal'].value
            data_length = len(raw_signals)
            batch_size = int(np.ceil(data_length/segment_length))
            raw_signals.resize(batch_size, segment_length, 1)
            sequence_length = np.array([segment_length for _ in raw_signals])
            sequence_length[-1] = data_length % segment_length
        if training:
            events = input_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
            s = events['start'] + events.attrs['read_start_rel_to_raw']
            indices = np.array([s//segment_length, s%segment_length])
            mask = np.unique(indices[0])
            transpose = np.cumsum(np.bincount(np.unique(mask))) - 1
            indices[0] = transpose[indices[0]]
            indices = indices.T
            #alphabet = [b'A', b'C', b'G', b'T', b'X']
            #values = list(map(lambda x:alphabet.index(x), events['base']))
            #values = np.argmax([events['base']==b'A',events['base']==b'C',events['base']==b'G',events['base']==b'T'],axis=0)
            values = np.sum([events['base']==b'C',
                             (events['base']==b'G')*2,
                             (events['base']==b'T')*3,
                            ], axis=0)
            raw_signals = raw_signals[mask]
            sequence_length = sequence_length[mask]
        else:
            indices = np.array([])
            values = np.array([])
        return raw_signals, sequence_length, indices, values

class fast5batches():
    def __init__(self, batch_size=100, segment_length=200, fast5dir='./', training=False, test_ratio=.2):
        self.training = training
        self.batch_size = batch_size
        self.segment_length = segment_length
        self.fast5s = []
        self.fast5s_test = []
        if training:
            for f in tqdm(glob(fast5dir+'/**/*.fast5',recursive=True),'checking files'):
                with h5py.File(f,'r') as input_data:
                    try:
                        input_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
                        self.fast5s.append(f)
                    except KeyError as e:
                        pass
            shuffle(self.fast5s)
            self.fast5s_test = self.fast5s[:int(len(self.fast5s)*test_ratio)]
            self.fast5s = self.fast5s[int(len(self.fast5s)*test_ratio):]
            print('{} files have labels, {} for training and {} for testing'.format(len(self.fast5s)+len(self.fast5s_test),
                                                                                    len(self.fast5s),
                                                                                    len(self.fast5s_test)))
        else:
            self.fast5s = glob(fast5dir+'/**/*.fast5',recursive=True)
            print('{} files added for evaluation'.format(len(self.fast5s)))
        self.f5_gen = gen(self.fast5s, inf=training)
        self.f5_gen_test = gen(self.fast5s_test, inf=training)
        #print(self.fast5s)
        self.raw_signals = np.array([])
        self.sequence_lengths = np.array([])
        self.indices = np.array([[],[]])
        self.values = np.array([])
        self.file_infos = []
        
        self.raw_signals_test = np.array([])
        self.sequence_lengths_test = np.array([])
        self.indices_test = np.array([[],[]])
        self.values_test = np.array([])
        self.file_infos_test = []
        
    def next_batch(self, test=False):
        f5_gen = self.f5_gen_test if self.training and test else self.f5_gen
        raw_signals = self.raw_signals_test if self.training and test else self.raw_signals
        sequence_lengths = self.sequence_lengths_test if self.training and test else self.sequence_lengths
        indices = self.indices_test if self.training and test else self.indices
        values = self.values_test if self.training and test else self.values
        file_infos = self.file_infos_test if self.training and test else self.file_infos
        
        while len(raw_signals)<self.batch_size:
            try:
                file_path = next(f5_gen)
                
                temp = fast5_to_valnlab(file_path, self.segment_length, self.training)
                new_raw_signals, new_sequence_lengths, new_indices, new_values = temp
                
                new_indices.T[0] += len(raw_signals)

                raw_signals = np.concatenate((raw_signals, new_raw_signals),0) if raw_signals.size else new_raw_signals
                sequence_lengths = np.concatenate((sequence_lengths, new_sequence_lengths),0) if sequence_lengths.size else new_sequence_lengths
                indices = np.concatenate((indices, new_indices),0) if indices.size else new_indices
                values = np.concatenate((values, new_values),0) if values.size else new_values
                file_infos += [{'name':file_path, 'i':i, 'total':len(new_raw_signals)} for i in range(len(new_raw_signals))]

            except StopIteration:
                if len(raw_signals)==0:
                    return
                else:
                    break
        
        return_raw_signals = raw_signals[:self.batch_size]
        return_sequence_lengths = sequence_lengths[:self.batch_size]
        return_values = values[indices.T[0]<self.batch_size]
        return_indices = indices[indices.T[0]<self.batch_size]
        return_file_infos = file_infos[:self.batch_size]
        
        if self.training and test:
            self.raw_signals_test = raw_signals[self.batch_size:]
            self.sequence_lengths_test = sequence_lengths[self.batch_size:]
            self.values_test = values[indices.T[0]>=self.batch_size]
            self.indices_test = indices[indices.T[0]>=self.batch_size]
            self.file_infos_test = file_infos[self.batch_size:]
            
            self.indices_test.T[0] -= self.batch_size
        else:
            self.raw_signals = raw_signals[self.batch_size:]
            self.sequence_lengths = sequence_lengths[self.batch_size:]
            self.values = values[indices.T[0]>=self.batch_size]
            self.indices = indices[indices.T[0]>=self.batch_size]
            self.file_infos = file_infos[self.batch_size:]
            
            self.indices.T[0] -= self.batch_size
        
        return return_raw_signals, return_sequence_lengths, return_values, return_indices, return_file_infos
        
