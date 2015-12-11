"""
Script: RSA300 Streamed Data File Parser
Date: 11/2014
Author: Morgan Allison
Software: Anaconda 2.1.0 (Python 2.7.6, 64-bit) http://continuum.io/downloads
Description: This script reads in a .r3f file created by the RSA306,
parses out all the metadata, saves the raw data, converts to IQ data, and exports
a *.mat file that is readable by SignalVu-PC

Directions: 
1. Save a streamed data file from your RSA300
2. Put it in C:\SignalVu-PC Files\
3. Run the script, enter the name of your source .r3f file 
and destination .mat file
4. Choose display features
5. Choose to save or discard footer

Features to add: Apply correction to ADC samples
"""

import os
from struct import *
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt

"""##########################Classes##########################"""

class VersionInfo:
	def __init__(self):
		self.fileid = []
		self.endian = []
		self.fileformatversion = []
		self.apiversion = []
		self.fx3version = []
		self.fpgaversion = []
		self.devicesn = []

class InstrumentState:
	def __init__(self):
		self.referencelevel = []
		self.centerfrequency = []
		self.temperature = []
		self.alignment = []
		self.freqreference = []
		self.trigmode = []
		self.trigsource = []
		self.trigtrans = []
		self.triglevel = []

class DataFormat:
	def __init__(self):
		self.datatype = []
		self.frameoffset = []
		self.framesize = []
		self.sampleoffset = []
		self.samplesize = []
		self.nonsampleoffset = []
		self.nonsamplesize = []
		self.ifcenterfrequency = []
		self.samplerate = []
		self.bandwidth = []
		self.corrected = []
		self.timetype = []
		self.reftime = []
		self.timesamples = []
		self.timesamplerate = []

class ChannelCorrection:
	def __init__(self):
		self.adcscale = []
		self.pathdelay = []
		self.correctiontype = []
		self.tableentries = []
		self.freqtable = []
		self.amptable = []
		self.phasetable = []

class FooterClass:
	def __init__(self):
		self.frame_descr = []
		self.frame_id = []
		self.trigger2_idx = []
		self.trigger1_idx = []
		self.time_sync_idx = []
		self.frame_status = []
		self.timestamp = []

class R3F:
	def __init__(self):
		base_directory = 'C:\\SignalVu-PC Files\\'
		ifilename = raw_input(
			'Enter input file name including extension (.r3f/.r3a/.r3h).\n> ')
		self.infile = base_directory + ifilename
		self.outfile = base_directory + ifilename[:-4]

		md_instructions = ("0=display nothing\n1=display header data" +
		"\n2=plot correction tables\n3=display 1 and 2\n> ")
		self.disp_flag = raw_input(md_instructions)
		
		self.iq_flag = raw_input("0=discard IQ\n1=save IQ in .mat file\n>")

		if '.r3h' in self.infile or '.r3a' in self.infile:
			print('\nBecause a .r3f file was not chosen, ' +
				'footer data cannot be extracted.\n')
			self.footer_flag = '0'
		else:
			self.footer_flag = raw_input('0=discard footer\n1=save footer\n> ')

		self.ADC = []
		self.IQ = []
		self.footer = []
		self.vinfo = VersionInfo()
		self.inststate = InstrumentState()
		self.dformat = DataFormat()
		self.chcorr = ChannelCorrection()

	def display_control(self):
		# Depending on the status of 'display,' display metadata, 
		# correction plots, both, or neither
		if self.disp_flag == '3':
			self.print_metadata()
			print('Metadata printed and channel correction graphs plotted.\n')
			self.plot_graphs()
		elif self.disp_flag == '2':
			print('Channel correction graphs plotted.\n')
			self.plot_graphs()
		elif self.disp_flag == '1':
			self.print_metadata()
			print('Metadata parsed and printed.\n')
		elif self.disp_flag== '0':
			print('Data parsed.')
		else: 
			print('Invalid choice for \'metadisplay\' variable. Select 0, 1, 2, or 3.')

	def get_header_data(self):
		# This function parses the header section of *.r3f and *.r3h files
		# Certain fields use np.fromstring() rather than unpack()
		# np.fromstring() allows the user to specify data type
		# unpack() saves data as a tuple, which can be used for printing
		# but not calculations
		if '.r3a' in self.infile or '.r3h' in self.infile:
			self.infile = self.infile[:-1] + 'h'
			print('\nYou specified a \'raw\' file format.\n' +
				'Header data is contained in .r3h files.\n' +
				'I\'ve loaded the .r3h file.\n')
		try:
			data = open(self.infile, 'rb').read(16384)
		except IOError:
			print('\nCannot open file. Check the input file name and try again.\n')
			quit()

		# Get and print File ID and Version Info sections of the header
		self.vinfo.fileid = data[:27]
		self.vinfo.endian = np.fromstring(
			data[512:516], dtype=np.uint32)
		self.vinfo.fileformatversion = unpack('4B', data[516:520])
		self.vinfo.apiversion = unpack('4B', data[520:524])
		self.vinfo.fx3version = unpack('4B', data[524:528])
		self.vinfo.fpgaversion = unpack('4B', data[528:532])
		self.vinfo.devicesn = data[532:596]
		
		# Get and print the Instrument State section of the header
		self.inststate.referencelevel = unpack('1d', data[1024:1032])
		self.inststate.centerfrequency = unpack('1d', data[1032:1040])
		self.inststate.temperature = unpack('1d', data[1040:1048])
		self.inststate.alignment = unpack('1I', data[1048:1052])
		self.inststate.freqreference = unpack('1I', data[1052:1056])
		self.inststate.trigmode = unpack('1I', data[1056:1060])
		self.inststate.trigsource = unpack('1I', data[1060:1064])
		self.inststate.trigtrans = unpack('1I', data[1064:1068])
		self.inststate.triglevel = unpack('1d', data[1068:1076])

		# Get and print Data Format section of the header
		self.dformat.datatype = unpack('1I', data[2048:2052])
		self.dformat.frameoffset = np.fromstring(
			data[2052:2056], dtype=np.uint32)
		self.dformat.framesize = np.fromstring(
			data[2056:2060], dtype=np.uint32)
		self.dformat.sampleoffset = unpack('1I', data[2060:2064])
		self.dformat.samplesize = np.fromstring(
			data[2064:2068], dtype=np.int32)
		self.dformat.nonsampleoffset = np.fromstring(
			data[2068:2072], dtype=np.uint32)
		self.dformat.nonsamplesize = np.fromstring(
			data[2072:2076], dtype=np.uint32)
		self.dformat.ifcenterfrequency = np.fromstring(
			data[2076:2084], dtype=np.double)
		self.dformat.samplerate = unpack('1d', data[2084:2092])
		self.dformat.bandwidth = unpack('1d', data[2092:2100])
		self.dformat.corrected = unpack('1I', data[2100:2104])
		self.dformat.timetype = unpack('1I', data[2104:2108])
		self.dformat.reftime = unpack('7i', data[2108:2136])
		self.dformat.timesamples = unpack('1Q', data[2136:2144])
		self.dformat.timesamplerate = np.fromstring(
			data[2144:2152], dtype=np.uint64)

		# Get Signal Path and Channel Correction data
		self.chcorr.adcscale = np.fromstring(
			data[3072:3080], dtype=np.double)
		self.chcorr.pathdelay = np.fromstring(
			data[3080:3088], dtype=np.double)
		self.chcorr.correctiontype = np.fromstring(
			data[4096:4100], dtype=np.uint32)
		tableentries = np.fromstring(data[4352:4356], dtype=np.uint32)
		self.chcorr.tableentries = tableentries	#purely for use in IQ_correction()
		freqindex = 4356
		phaseindex = freqindex + 501*4
		ampindex = phaseindex + 501*4
		self.chcorr.freqtable = np.fromstring(
			data[freqindex:(freqindex+tableentries*4)], dtype=np.float32)
		self.chcorr.phasetable = np.fromstring(
			data[phaseindex:(phaseindex+tableentries*4)], dtype=np.float32)
		self.chcorr.amptable = np.fromstring(
			data[ampindex:(ampindex+tableentries*4)], dtype=np.float32)
		
	def print_metadata(self):
		# This function simply prints out all the metadata contained in the header
		print('FILE INFO')
		print('FileID: %s' % self.vinfo.fileid)
		print('Endian Check: 0x%x' % self.vinfo.endian)
		print('File Format Version: %i.%i.%i.%i' % self.vinfo.fileformatversion)
		print('API Version: %i.%i.%i.%i' % self.vinfo.apiversion)
		print('FX3 Version: %i.%i.%i.%i' % self.vinfo.fx3version)
		print('FPGA Version: %i.%i.%i.%i' % self.vinfo.fpgaversion)
		print('Device Serial Number: %s' % self.vinfo.devicesn)

		print('INSTRUMENT STATE')
		print('Reference Level: %d dBm' % self.inststate.referencelevel)
		print('Center Frequency: %d Hz' % self.inststate.centerfrequency)
		print('Temperature: %d C' % self.inststate.temperature)
		print('Alignment status: %d' % self.inststate.alignment)
		print('Frequency Reference: %d' % self.inststate.freqreference)
		print('Trigger mode: %d' % self.inststate.trigmode)
		print('Trigger Source: %d' % self.inststate.trigsource)
		print('Trigger Transition: %d' % self.inststate.trigtrans)
		print('Trigger Level: %d dBm\n' % self.inststate.triglevel)

		print('DATA FORMAT')
		print('Data Type: %i' % self.dformat.datatype)
		print('Offset to first frame (bytes): %i' % self.dformat.frameoffset)
		print('Frame Size (bytes): %i' % self.dformat.framesize)
		print('Offset to sample data (bytes): %i' % self.dformat.sampleoffset)
		print('Samples in Frame: %i' % self.dformat.samplesize)
		print('Offset to non-sample data: %i' % self.dformat.nonsampleoffset)
		print('Size of non-sample data: %i' % self.dformat.nonsamplesize)
		print('IF Center Frequency: %i Hz' % self.dformat.ifcenterfrequency)
		print('Sample Rate: %i S/sec' % self.dformat.samplerate)
		print('Bandwidth: %i Hz' % self.dformat.bandwidth)
		print('Corrected data status: %i' % self.dformat.corrected)
		print('Time Type (0=local, 1=remote): %i' % self.dformat.timetype)
		print('Reference Time: %i %i/%i at %i:%i:%i:%i' % self.dformat.reftime)
		print('Sample count: %i' % self.dformat.timesamples)
		print('Sample ticks per second: %i\n' % self.dformat.timesamplerate)

		print('CHANNEL AND SIGNAL PATH CORRECTION')
		print('ADC Scale Factor: %12.12f' % self.chcorr.adcscale)
		print('Signal Path Delay: %f nsec' % (self.chcorr.pathdelay*1e9))
		print('Correction Type (0=LF, 1=IF): %i\n' % self.chcorr.correctiontype)

	def plot_graphs(self):
		# This function plots the amplitude and phase correction 
		# tables as a function of IF frequency
		plt.subplot(2,1,1)
		plt.plot(self.chcorr.freqtable/1e6,self.chcorr.amptable)
		plt.title('Amplitude and Phase Correction')
		plt.ylabel('Amplitude (dB)')
		plt.subplot(2,1,2)
		plt.plot(self.chcorr.freqtable/1e6,self.chcorr.phasetable)
		plt.ylabel('Phase (degrees)')
		plt.xlabel('IF Frequency (MHz)')
		plt.show()
		plt.clf()

	def get_adc_samples(self):
		if '.r3a' in self.infile or '.r3h' in self.infile:
			self.infile = self.infile[:-1] + 'a'
			print('\nYou specified a \'raw\' file format.\n' +
				'Sample data is contained in .r3a files.\n' +
				'I\'ve loaded the .r3a file.\n')
		try:
			data = open(self.infile, 'rb')
		except IOError:
			print('\nCannot open file. Check the input file name and try again.\n')
			quit()
		filesize = os.path.getsize(self.infile)

		#Filter file type and read the file appropriately
		if '.r3f' in self.infile:
			self.numframes = (filesize/self.dformat.framesize) - 1
			print('Number of Frames: {}\n'.format(self.numframes))
			data.seek(self.dformat.frameoffset)
			adcdata = np.empty(self.numframes*self.dformat.samplesize)
			fstart = 0
			fstop = self.dformat.samplesize
			self.footer = range(0,self.numframes)
			for i in range(self.numframes):
				frame = data.read(self.dformat.nonsampleoffset)
				adcdata[fstart:fstop] = np.fromstring(frame, dtype=np.int16)
				fstart = fstop
				fstop = fstop + self.dformat.samplesize
				if self.footer_flag == '0':
					data.seek(self.dformat.nonsamplesize,1)
				else:
					temp_ftr = data.read(self.dformat.nonsamplesize)
					self.footer[i] = FooterClass()
					self.footer[i] = self.parse_footer(temp_ftr)
		elif '.r3a' in self.infile:
			adcdata = np.fromfile(self.infile, dtype=np.int16)
		else:
			print('Invalid file type. Please specify a .r3f, .r3a, or .r3h file.\n')

		#Scale ADC data
		self.ADC = adcdata*self.chcorr.adcscale

	def parse_footer(self, raw_footer):
		footer = FooterClass()
		footer.reserved = np.fromstring(raw_footer[0:6], dtype=np.uint16, count=3)
		footer.frame_id = np.fromstring(raw_footer[8:12], dtype=np.uint32, count=1)
		footer.trigger2_idx = np.fromstring(raw_footer[12:14], dtype=np.uint16, count=1)
		footer.trigger1_idx = np.fromstring(raw_footer[14:16], dtype=np.uint16, count=1)
		footer.time_sync_idx = np.fromstring(raw_footer[16:18], dtype=np.uint16, count=1)
		footer.frame_status = '{0:b}'.format(int(np.fromstring(raw_footer[18:20], 
			dtype=np.uint16, count=1)))
		footer.timestamp = np.fromstring(raw_footer[20:28], dtype=np.uint64, count=1)
		
		return footer

	def ddc(self):
		if self.iq_flag =='1':
			#Generate quadrature signals
			IFfreq = self.dformat.ifcenterfrequency
			size = len(self.ADC)
			sampleperiod = 1.0/self.dformat.timesamplerate
			xaxis = np.linspace(0,size*sampleperiod,size)
			LO_I = np.sin(IFfreq*(2*np.pi)*xaxis)
			LO_Q = np.cos(IFfreq*(2*np.pi)*xaxis)

			#Run ADC data through digital down converter
			I = self.ADC*LO_I
			Q = self.ADC*LO_Q
			nyquist = self.dformat.timesamplerate/2
			cutoff = 40e6/nyquist
			IQfilter = signal.firwin(300, cutoff, window=('kaiser', 10))
			I = signal.lfilter(IQfilter, 1.0, I)
			Q = signal.lfilter(IQfilter, 1.0, Q)
			IQ = I + 1j*Q
			IQ = 2*IQ
			print('Digital Down Conversion Complete.\n')
			self.IQ = IQ

	def save_mat_file(self):
		if self.iq_flag == '1':
			InputCenter = self.inststate.centerfrequency
			XDelta = 1.0/self.dformat.timesamplerate
			Y = self.IQ
			InputZoom = np.uint8(1)
			Span = self.dformat.bandwidth
			sio.savemat(self.outfile, {'InputCenter':InputCenter,'Span':Span,'XDelta':XDelta,
				'Y':Y,'InputZoom':InputZoom}, format='5')
			print('Data file saved at {}.mat'.format(self.outfile))

	def save_footer_file(self):
		if self.footer_flag == '1':
			fname = self.outfile + '.txt'
			ffile = open(fname, 'w')
			ffile.write('FrameID\tTrig1\tTrig2\tTSync\tFrmStatus\tTimeStamp\n')
			for i in range(self.numframes):
				ffile.write(', '.join(map(str, self.footer[i].frame_id)))
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].trigger2_idx)))
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].trigger1_idx)))
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].time_sync_idx)))
				ffile.write('\t')
				ffile.write(self.footer[i].frame_status)
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].timestamp)))
				ffile.write('\n')
			ffile.close()
			print('Footer file saved at {}.\n'.format(fname))

	"""
	def IQ_correction(IQ, metadata):
		#amp and phase correction filter NOT FINISHED
		#CorrectionFrameSize
		framesize = self.chcorr.tableentries
		amptable = self.chcorr.amptable
		phasetable = self.chcorr.phasetable
		#Convert magnitude from dB to V and phase from degrees to rad
		amptable = 1/np.sqrt(10**(amptable/(2**15*10)))
		phasetable = phasetable*np.pi/180
		correct_real = amptable*np.cos(np.radians(phasetable))
		correct_imag = amptable*np.sin(np.radians(phasetable))
		correctFD = correct_real + 1j*correct_imag
		corr_frames = len(self.IQ)/framesize
		IQi = 0
		
		#Apply correction factors to IQ data frame by frame
		for frame in range(0,corr_frames):
			cframe = self.IQ[IQi:IQi+framesize]
			cframe = fft(cframe)
			cframe = cframe*correctFD
			cframe = ifft(cframe)
			self.IQ[IQi:IQi+framesize] = cframe
			IQi = IQi+framesize
		print('IQ Correction Applied.')
	"""	

def main():
	r3f = R3F()
	r3f.get_header_data()
	r3f.display_control()
	r3f.get_adc_samples()
	r3f.ddc()
	r3f.save_mat_file()
	r3f.save_footer_file()
	
if __name__ == '__main__':
	main()