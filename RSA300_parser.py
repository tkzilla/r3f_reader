"""
Script: RSA300 Streamed Data File Parser
Date: 12/2015
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
5. Choose to save or discard IQ data
6. Choose to save or discard footer (iff .r3f file)

TODO:
2. Add .r3a functionality to file_muncher()
3. Figure out a coherent control method and function structure
so users can use the .r3f class without having to be you
"""

import os
from struct import *
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import math

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
		self.clocksamples = []
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
		fname = raw_input(
			'Enter input file name including extension (.r3f/.r3a/.r3h).\n> ')
		self.infilename = base_directory + fname
		self.outfile = base_directory + fname[:-4]

		md_instructions = ("0=display nothing\n1=display header data" +
		"\n2=plot correction tables\n3=display 1 and 2\n> ")
		self.disp_flag = raw_input(md_instructions)
		
		self.iq_flag = raw_input("0=discard IQ\n1=save IQ in .mat file\n>")

		if '.r3h' in self.infilename or '.r3a' in self.infilename:
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
		# Depending on the status of 'display,' display header data, 
		# correction plots, both, or neither
		if self.disp_flag == '3':
			self.print_metadata()
			print('Header printed and channel correction graphs plotted.\n')
			self.plot_graphs()
		elif self.disp_flag == '2':
			print('Channel correction graphs plotted.\n')
			self.plot_graphs()
		elif self.disp_flag == '1':
			self.print_metadata()
			print('Header printed.\n')
		elif self.disp_flag== '0':
			print('Header parsed.')
		else: 
			print('Invalid choice for \'metadisplay\' variable. Select 0, 1, 2, or 3.')

	def get_header_data(self):
		# Parses the header section of *.r3f and *.r3h files.
		# Certain fields use np.fromstring() rather than unpack()
		# np.fromstring() allows the user to specify data type
		# unpack() saves data as a tuple, which can be used for printing
		# but not calculations
		if '.r3a' in self.infilename or '.r3h' in self.infilename:
			self.infilename = self.infilename[:-1] + 'h'
			print('You specified a \'raw\' file format.\n' +
				'Header data is contained in .r3h files.\n' +
				'I\'ve loaded the .r3h file to get header data.\n')
		try:
			self.infile = open(self.infilename, 'rb')
			data = self.infile.read(16384)
		except IOError:
			print('\nCannot open file. Check the input file name and try again.\n')
			quit()

		# Get File ID and Version Info sections of the header
		self.vinfo.fileid = data[:27]
		self.vinfo.endian = np.fromstring(
			data[512:516], dtype=np.uint32)
		self.vinfo.fileformatversion = unpack('4B', data[516:520])
		self.vinfo.apiversion = unpack('4B', data[520:524])
		self.vinfo.fx3version = unpack('4B', data[524:528])
		self.vinfo.fpgaversion = unpack('4B', data[528:532])
		self.vinfo.devicesn = data[532:596]
		
		# Get the Instrument State section of the header
		self.inststate.referencelevel = unpack('1d', data[1024:1032])
		self.inststate.centerfrequency = unpack('1d', data[1032:1040])
		self.inststate.temperature = unpack('1d', data[1040:1048])
		self.inststate.alignment = unpack('1I', data[1048:1052])
		self.inststate.freqreference = unpack('1I', data[1052:1056])
		self.inststate.trigmode = unpack('1I', data[1056:1060])
		self.inststate.trigsource = unpack('1I', data[1060:1064])
		self.inststate.trigtrans = unpack('1I', data[1064:1068])
		self.inststate.triglevel = unpack('1d', data[1068:1076])

		# Get Data Format section of the header
		#self.dformat.datatype = unpack('1I', data[2048:2052])
		self.dformat.datatype = np.fromstring(
			data[2048:2052], dtype=np.uint32)
		if self.dformat.datatype == 161:
			self.dformat.datatype = 2 #bytes per sample
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
		#self.dformat.samplerate = unpack('1d', data[2084:2092])
		self.dformat.samplerate = np.fromstring(
			data[2084:2092], dtype=np.double)
		self.dformat.bandwidth = unpack('1d', data[2092:2100])
		self.dformat.corrected = unpack('1I', data[2100:2104])
		self.dformat.timetype = unpack('1I', data[2104:2108])
		self.dformat.reftime = unpack('7i', data[2108:2136])
		self.dformat.clocksamples = unpack('1Q', data[2136:2144])
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
		self.infile.close()
		
	def print_metadata(self):
		# This function simply prints out all the header data
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
		if self.dformat.datatype == 161:
			self.dformat.datatype = 2 	#bytes per sample
		print('Data Type: %i bytes per sample' % self.dformat.datatype)
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
		print('Clock sample count: %i' % self.dformat.clocksamples)
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

	def file_duration_check(self):
		# Opens the input file and checks its size (bytes) and duration (sec)
		try:
			self.infile = open(self.infilename, 'rb')
		except IOError:
			print('\nCannot open file. Check the input file name and try again.\n')
			quit()
		self.filesize = os.path.getsize(self.infilename)
		if '.r3a' in self.infilename or '.r3h' in self.infilename:
			self.infilename = self.infilename[:-1] + 'a'
			print('\nYou specified a \'raw\' file format.\n' +
				'Sample data is contained in .r3a files.\n' +
				'I\'ve loaded the .r3a file to get ADC data.\n')

			self.filelength = self.filesize/self.dformat.datatype/self.dformat.samplerate
			print('File size is: {} bytes.\nFile Length is {} seconds.\n'.format(
				self.filesize, self.filelength))
		elif '.r3f' in self.infilename:
			self.numframes = int((self.filesize/self.dformat.framesize) - 1)
			#because it needs to be unsigned to check looping in file_muncher correctly
			self.filelength = self.numframes*(self.dformat.samplesize/self.dformat.samplerate)
			print('Number of Frames: {}\n'.format(self.numframes))
			print('File size is: {} bytes.'.format(self.filesize))
			print('File length is: {} seconds.\n'.format(self.filelength))

	def file_muncher(self):
		# Determines if the file is longer than 1 second and splits accordingly
		# This function calls get_adc_samples(), ddc(), and save_mat_file(),
		# which are really the core processing components of this script
		loop = 0
		if '.r3f' in self.infilename:
			fps = 13698
			while self.numframes > 0:
				if self.numframes > fps:
					print('File is longer than 1 second, splitting into {} files.'
						.format(int(math.ceil(self.filelength))))
					process_frames = fps
				else:
					process_frames = self.numframes
				if loop == 0:
					startpoint = self.dformat.frameoffset
				elif loop > 0:
					startpoint = loop*fps*self.dformat.framesize
				self.get_adc_samples(process_frames, startpoint)
				#print('Beginning DDC.')
				#start = time.clock()
				self.ddc()
				#end = time.clock()
				#print('DDC processing time: {}'.format(end-start))
				#secondspersecond = (end-start)/self.filelength
				#print('Seconds of DDC processing per second of file length: {}'.format(secondspersecond))
				self.save_mat_file(loop)
				self.numframes -= fps
				loop += 1
		#elif '.r3a' in self.infilename:

	def get_adc_samples(self, process_frames, startpoint):
		# Reads ADC samples from input file 
		# Skips over or saves footer for .r3f files
		# Reads in everything for .r3a files

		#Filter file type and read the file appropriately
		if '.r3f' in self.infilename:
			self.infile.seek(startpoint)
			adcsamples = np.empty(process_frames*self.dformat.samplesize)
			fstart = 0
			fstop = self.dformat.samplesize
			self.footer = range(process_frames)
			for i in xrange(process_frames):
				frame = self.infile.read(self.dformat.nonsampleoffset)
				adcsamples[fstart:fstop] = np.fromstring(frame, dtype=np.int16)
				fstart = fstop
				fstop = fstop + self.dformat.samplesize
				if self.footer_flag == '0':
					self.infile.seek(self.dformat.nonsamplesize,1)
				else:
					temp_ftr = self.infile.read(self.dformat.nonsamplesize)
					self.footer[i] = FooterClass()
					self.footer[i] = self.parse_footer(temp_ftr)
		elif '.r3a' in self.infilename:
			adcsamples = np.fromfile(self.infilename, dtype=np.int16)
		else:
			print('Invalid file type. Please specify a .r3f, .r3a, or .r3h file.\n')

		#Scale ADC data
		self.ADC = adcsamples*self.chcorr.adcscale

	def parse_footer(self, raw_footer):
		# Parses footer based on internal footer documentation
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
		# Digital downconverter that converts ADC data to IQ data
		if self.iq_flag =='1':
			#Generate quadrature signals
			IFfreq = self.dformat.ifcenterfrequency
			size = len(self.ADC)
			sampleperiod = 1.0/self.dformat.timesamplerate
			xaxis = np.linspace(0,size*sampleperiod,size)
			LO_I = np.sin(IFfreq*(2*np.pi)*xaxis)
			LO_Q = np.cos(IFfreq*(2*np.pi)*xaxis)
			del(xaxis)

			#Run ADC data through digital downconverter
			I = self.ADC*LO_I
			Q = self.ADC*LO_Q
			del(LO_I)
			del(LO_Q)
			nyquist = self.dformat.timesamplerate/2
			cutoff = 40e6/nyquist
			IQfilter = signal.firwin(32, cutoff, window=('kaiser', 2.23))
			I = signal.lfilter(IQfilter, 1.0, I)
			Q = signal.lfilter(IQfilter, 1.0, Q)
			IQ = I + 1j*Q
			IQ = 2*IQ
			self.IQ = IQ

	def save_mat_file(self, loop):
		# Saves a .mat file containing variables specified in the 
		# SignalVu-PC help file
		if self.iq_flag == '1':
			InputCenter = self.inststate.centerfrequency
			XDelta = 1.0/self.dformat.timesamplerate
			Y = self.IQ
			InputZoom = np.uint8(1)
			Span = self.dformat.bandwidth
			filename = self.outfile + '_' + str(loop)
			sio.savemat(filename, {'InputCenter':InputCenter,'Span':Span,'XDelta':XDelta,
				'Y':Y,'InputZoom':InputZoom}, format='5')
			print('Data file saved at {}.mat'.format(filename))

	def save_footer_file(self):
		# Saves footer data in a .txt file
		if self.footer_flag == '1':
			fname = self.outfile + '.txt'
			ffile = open(fname, 'w')
			ffile.write('FrameID\tTrig1\tTrig2\tTSync\tFrmStatus\tTimeStamp\n')
			for i in xrange(self.numframes):
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
		for frame in xrange(0,corr_frames):
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
	r3f.file_duration_check()
	r3f.file_muncher()
	"""
	r3f.get_adc_samples()
	print('starting ddc')
	start = time.clock()
	r3f.ddc()
	end = time.clock()
	print('ddc complete')
	print('time: {}'.format(end-start))
	secondspersecond = (end-start)/r3f.filelength
	print('Seconds of DDC processing per second of file length: {}'.format(secondspersecond))
	r3f.save_mat_file()
	r3f.save_footer_file()
	"""

if __name__ == '__main__':
	main()