from random import shuffle
from glob import glob
from tqdm import tqdm
from statsmodels import robust
import numpy as np
import h5py
from scipy.sparse import csr_matrix, vstack
from collections import namedtuple
import time


def gen(l):
    n = 0
    while True:
        for ll in l:
            yield n, ll
        n += 1
            
def fast5_to_valnlab(file_path, segment_length, training=True, overlap=0):
    with h5py.File(file_path,'r') as input_data:
        
        #digitisation = input_data['/UniqueGlobalKey/channel_id'].attrs['digitisation']
        #offset = input_data['/UniqueGlobalKey/channel_id'].attrs['offset']
        #range_ = input_data['/UniqueGlobalKey/channel_id'].attrs['range']
        #heatsink_temp = float(input_data['/UniqueGlobalKey/tracking_id'].attrs['heatsink_temp'].decode())
        #asic_temp = float(input_data['/UniqueGlobalKey/tracking_id'].attrs['asic_temp'].decode())
        sampling_rate = input_data['/UniqueGlobalKey/channel_id'].attrs['sampling_rate']
        
        for read_name in input_data['Raw/Reads']:
            start_time = input_data['/Raw/Reads'][read_name].attrs['start_time']
            median_before = input_data['/Raw/Reads'][read_name].attrs['median_before']
            raw_signals = input_data['Raw/Reads'][read_name]['Signal'].value
        
        median = np.median(raw_signals)
        raw_signals = (raw_signals - median) / np.median(np.abs(raw_signals - median))
        data_length = len(raw_signals)
        batch_size = np.ceil(data_length / (segment_length - overlap)).astype(int)
        
        ix = np.mgrid[:batch_size, :segment_length][1] 
        ix += np.arange(batch_size, dtype=int)[:,None] * (segment_length - overlap)

        raw_time = start_time/1000 + np.arange(data_length)/sampling_rate
        raw_time = raw_time / (48 * 60 * 60)

        mb = np.ones(data_length) * (median_before/median)

        features = [raw_signals, raw_time, mb]
        #features = [raw_signals]
        values = np.vstack(features).T
        values = np.resize(values, (np.max(ix)+1, len(features)))[ix]
        sequence_lengths = np.sum(ix<data_length, axis=1)
        
        labels = None
        if training:
            events = input_data['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
            s = events['start'] + events.attrs['read_start_rel_to_raw']
            b = events['base']
            intbases = 1 + np.sum([b==b'C', (b==b'G')*2, (b==b'T')*3], axis=0) 
            labels = csr_matrix((intbases, (np.zeros_like(s),s)), shape=(1,np.max(ix)+1))[0,ix]
            
            mask = labels.getnnz(1) > 0
            
            labels = labels[mask]
            sequence_lengths = sequence_lengths[mask]
            values = values[mask]
        return values, sequence_lengths, labels#, values#, meta

class fast5batches():
    def __init__(self, batch_size=100, segment_length=200, fast5dir='./', training=False, test_ratio=.2, overlap=0):
        self.training = training
        self.batch_size = batch_size
        self.segment_length = segment_length
        self.overlap = overlap
        self.fast5s = []
        self.fast5s_test = []
        len_longes_filename = 0
        if training:
            for f in tqdm(glob(fast5dir+'/**/*.fast5',recursive=True),'checking files'):
                if len(f)>len_longes_filename:
                    len_longes_filename = len(f)
                try:
                    with h5py.File(f,'r') as input_data:
                        if ('RawGenomeCorrected_000' in input_data['Analyses'] and
                            'BaseCalled_template' in input_data['Analyses/RawGenomeCorrected_000']):
                            self.fast5s.append(f)
                except (KeyError, OSError) as e:
                    pass
            shuffle(self.fast5s)
            self.fast5s_test = self.fast5s[:int(len(self.fast5s)*test_ratio)]
            self.fast5s = self.fast5s[int(len(self.fast5s)*test_ratio):]
            print('{} files have labels, {} for training and {} for testing'.format(len(self.fast5s)+len(self.fast5s_test),
                                                                                    len(self.fast5s),
                                                                                    len(self.fast5s_test)))
        else:
            self.fast5s = glob(fast5dir+'/**/*.fast5', recursive=True)
            len_longes_filename = max(len(f) for f in self.fast5s)
            print('{} files added for evaluation'.format(len(self.fast5s)))
        
        self.f5_gen = gen(self.fast5s)
        self.f5_gen_test = gen(self.fast5s_test)
        self.dtype_fileinfos = [('name','S{}'.format(len_longes_filename)),('i','uint32'),('total','uint32')]
        
        self.raw_signals = np.array([])
        self.sequence_lengths = np.array([])
        self.labels = None
        self.file_infos = np.array([], dtype=self.dtype_fileinfos)
        
        self.raw_signals_test = np.array([])
        self.sequence_lengths_test = np.array([])
        self.labels_test = None
        self.file_infos_test = np.array([], dtype=self.dtype_fileinfos)
        

        
    def next_batch(self, test=False, fill=False):
        
        f5_gen = self.f5_gen_test if self.training and test else self.f5_gen
        raw_signals = self.raw_signals_test if self.training and test else self.raw_signals
        sequence_lengths = self.sequence_lengths_test if self.training and test else self.sequence_lengths
        labels = self.labels_test if self.training and test else self.labels
        file_infos = self.file_infos_test if self.training and test else self.file_infos
        
        nt = namedtuple('Labels', ['indices', 'values'])
        if fill:
            bar = tqdm(desc='filling {} cache'.format({True:'testing', False:'training'}[test]))
        while not self.training and len(raw_signals)<self.batch_size or \
              self.training and len(np.unique(file_infos['name'])) < self.batch_size*1:
            epoch, file_path = next(f5_gen)
            
            if not self.training and epoch>0:
                if len(raw_signals)==0:
                    return
                break
            temp = fast5_to_valnlab(file_path, self.segment_length, self.training, overlap=self.overlap)
            new_raw_signals, new_sequence_lengths, new_labels = temp
            
            new_file_infos = np.zeros(len(new_raw_signals), dtype=self.dtype_fileinfos)
            new_file_infos['name'] = file_path
            new_file_infos['i'] = np.arange(len(new_raw_signals))
            new_file_infos['total'] = len(new_raw_signals)
            

            raw_signals = np.concatenate((raw_signals, new_raw_signals),0) if raw_signals.size else new_raw_signals
            sequence_lengths = np.concatenate((sequence_lengths, new_sequence_lengths),0) if sequence_lengths.size else new_sequence_lengths
            labels = vstack([labels, new_labels]) if labels is not None else new_labels
            file_infos = np.concatenate((file_infos, new_file_infos),0) if file_infos.size else new_file_infos
            
            if fill:
                bar.update()
        
        mask = np.zeros(raw_signals.shape[0], dtype=bool)
        if not fill:
            mask[:self.batch_size] = True
        else:
            bar.close()
        
        return_labels = None
        if self.training:
            np.random.shuffle(mask)
            coo = labels[mask].tocoo()
            rows, cols, data = coo.row, coo.col, coo.data
            ind = np.lexsort((cols, rows))
            return_labels = nt(indices=np.vstack([rows[ind],cols[ind]]).T, values=data[ind]-1)
        return_raw_signals = raw_signals[mask]
        return_sequence_lengths = sequence_lengths[mask]
        return_file_infos = file_infos[mask]
        
            
        
        if self.training and test:
            self.raw_signals_test = raw_signals[~mask]
            self.sequence_lengths_test = sequence_lengths[~mask]
            self.labels_test = labels[~mask]
            self.file_infos_test = file_infos[~mask]
        else:
            self.raw_signals = raw_signals[~mask]
            self.sequence_lengths = sequence_lengths[~mask]
            self.labels = labels[~mask] if self.training else None
            self.file_infos = file_infos[~mask]
        
        return return_raw_signals, return_sequence_lengths, return_labels, return_file_infos
        
