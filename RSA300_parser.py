"""
Script: RSA300 Streamed Data File Parser
Date: 11/2014
Author: Morgan Allison
Software: Anaconda 2.1.0 (Python 2.7.6, 64-bit) http://continuum.io/downloads
Description: This script reads in a streamed data file created by the RSA300,
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
"""
versioninfo = {'fileid': fileid, 'endian': endian, 'fileformatversion': fileformatversion,
	'apiversion': apiversion, 'fx3version': fx3version, 'fpgaversion': fpgaversion, 'devicesn': devicesn}

instrumentstate = {'referencelevel': referencelevel, 'centerfrequency': centerfrequency,
		'temperature': temperature, 'alignment': alignment, 'freqreference': freqreference,
		'trigmode': trigmode, 'trigsource': trigsource, 'trigtrans': trigtrans,
		'triglevel': triglevel}

dataformat = {'datatype': datatype, 'frameoffset': frameoffset, 'framesize':framesize,
		'sampleoffset':sampleoffset, 'samplesize': samplesize, 'nonsampleoffset': nonsampleoffset,
		'nonsamplesize': nonsamplesize, 'ifcenterfrequency': ifcenterfrequency,
		'samplerate': samplerate, 'bandwidth': bandwidth, 'corrected': corrected,
		'timetype': timetype, 'reftime': reftime, 'timesamples': timesamples,
		'timesamplerate': timesamplerate}

channelcorrection = {'adcscale': adcscale, 'pathdelay': pathdelay, 
	'correctiontype':correctiontype, 'tableentries': tableentries, 
	'freqtable': freqtable, 'amptable': amptable, 'phasetable': phasetable}

metadata = {'versioninfo': versioninfo, 'instrumentstate': instrumentstate,
	'dataformat': dataformat, 'channelcorrection': channelcorrection}

footer = {'frame_descr': frame_descr, 'frame_id': frame_id,
	'trigger2_idx': trigger2_idx, 'trigger1_idx': trigger1_idx,
	'time_sync_idx': time_sync_idx, 'frame_status': frame_status,
	'timestamp': timestamp}
"""	
# This function simply prints out all the metadata contained in the header
def print_metadata(metadata):
	print('FILE INFO')
	print('FileID: %s' % metadata['versioninfo']['fileid'])
	print('Endian Check: 0x%x' % metadata['versioninfo']['endian'])
	print('File Format Version: %i.%i.%i.%i' % metadata['versioninfo']['fileformatversion'])
	print('API Version: %i.%i.%i.%i' % metadata['versioninfo']['apiversion'])
	print('FX3 Version: %i.%i.%i.%i' % metadata['versioninfo']['fx3version'])
	print('FPGA Version: %i.%i.%i.%i' % metadata['versioninfo']['fpgaversion'])
	print('Device Serial Number: %s' % metadata['versioninfo']['devicesn'])

	print('INSTRUMENT STATE')
	print('Reference Level: %d dBm' % metadata['instrumentstate']['referencelevel'])
	print('Center Frequency: %d Hz' % metadata['instrumentstate']['centerfrequency'])
	print('Temperature: %d C' % metadata['instrumentstate']['temperature'])
	print('Alignment status: %d' % metadata['instrumentstate']['alignment'])
	print('Frequency Reference: %d' % metadata['instrumentstate']['freqreference'])
	print('Trigger mode: %d' % metadata['instrumentstate']['trigmode'])
	print('Trigger Source: %d' % metadata['instrumentstate']['trigsource'])
	print('Trigger Transition: %d' % metadata['instrumentstate']['trigtrans'])
	print('Trigger Level: %d dBm\n' % metadata['instrumentstate']['triglevel'])

	print('DATA FORMAT')
	print('Data Type: %i' % metadata['dataformat']['datatype'])
	print('Offset to first frame (bytes): %i' % metadata['dataformat']['frameoffset'])
	print('Frame Size (bytes): %i' % metadata['dataformat']['framesize'])
	print('Offset to sample data (bytes): %i' % metadata['dataformat']['sampleoffset'])
	print('Samples in Frame: %i' % metadata['dataformat']['samplesize'])
	print('Offset to non-sample data: %i' % metadata['dataformat']['nonsampleoffset'])
	print('Size of non-sample data: %i' % metadata['dataformat']['nonsamplesize'])
	print('IF Center Frequency: %i Hz' % metadata['dataformat']['ifcenterfrequency'])
	print('Sample Rate: %i S/sec' % metadata['dataformat']['samplerate'])
	print('Bandwidth: %i Hz' % metadata['dataformat']['bandwidth'])
	print('Corrected data status: %i' % metadata['dataformat']['corrected'])
	print('Time Type (local = 0/remote = 1): %i' % metadata['dataformat']['timetype'])
	print('Reference Time: %i %i/%i at %i:%i:%i:%i' % metadata['dataformat']['reftime'])
	print('Sample count: %i' % metadata['dataformat']['timesamples'])
	print('Sample ticks per second: %i\n' % metadata['dataformat']['timesamplerate'])

	print('CHANNEL AND SIGNAL PATH CORRECTION')
	print('ADC Scale Factor: %12.12f' % metadata['channelcorrection']['adcscale'])
	print('Signal Path Delay: %f nsec' % (metadata['channelcorrection']['pathdelay']*1e9))
	print('Correction Type (0 = LF, 1 = IF): %i' % metadata['channelcorrection']['correctiontype'])
	print('Number of table entries: %i' % metadata['channelcorrection']['tableentries'])

# This function plots the amplitude and phase correction 
# tables as a function of IF frequency
def plot_graphs(metadata):
	plt.subplot(2,1,1)
	plt.plot(metadata['channelcorrection']['freqtable']/1e6,metadata['channelcorrection']['amptable'])
	plt.title('Amplitude and Phase Correction')
	plt.ylabel('Amplitude (dB)')
	plt.subplot(2,1,2)
	plt.plot(metadata['channelcorrection']['freqtable']/1e6,metadata['channelcorrection']['phasetable'])
	plt.ylabel('Phase (degrees)')
	plt.xlabel('IF Frequency (MHz)')
	plt.show()
	plt.clf()

# This function reads and parses the header section 
# of *.r3f and *.r3h files
# certain fields use np.fromstring() rather than unpack()
# np.fromstring() allows the user to specify data type
# unpack() saves data as a tuple, which can be used for printing
# but not calculations
def get_header_data(infile, display):
	try:
		datafile = open(infile, 'rb')
	except IOError:
		print('\nInvalid input file. Check the \'infile\' variable and try again.')
		quit()
	data = datafile.read(16384)

	# Get and print File ID and Version Info sections of the header
	fileid = data[:27]
	endian = np.fromstring(data[512:516], dtype=np.uint32)
	fileformatversion = unpack('4B', data[516:520])
	apiversion = unpack('4B', data[520:524])
	fx3version = unpack('4B', data[524:528])
	fpgaversion = unpack('4B', data[528:532])
	devicesn = data[532:596]
	versioninfo = {'fileid': fileid, 'endian': endian, 'fileformatversion': fileformatversion,
		'apiversion': apiversion, 'fx3version': fx3version, 'fpgaversion': fpgaversion, 'devicesn': devicesn}

	# Get and print the Instrument State section of the header
	referencelevel = unpack('1d', data[1024:1032])
	centerfrequency = unpack('1d', data[1032:1040])
	temperature = unpack('1d', data[1040:1048])
	alignment = unpack('1I', data[1048:1052])
	freqreference = unpack('1I', data[1052:1056])
	trigmode = unpack('1I', data[1056:1060])
	trigsource = unpack('1I', data[1060:1064])
	trigtrans = unpack('1I', data[1064:1068])
	triglevel = unpack('1d', data[1068:1076])
	instrumentstate = {'referencelevel': referencelevel, 'centerfrequency': centerfrequency,
		'temperature': temperature, 'alignment': alignment, 'freqreference': freqreference,
		'trigmode': trigmode, 'trigsource': trigsource, 'trigtrans': trigtrans,
		'triglevel': triglevel}

	# Get and print Data Format section of the header
	datatype = unpack('1I', data[2048:2052])
	frameoffset = np.fromstring(data[2052:2056], dtype=np.uint32)
	framesize = np.fromstring(data[2056:2060], dtype=np.uint32)
	sampleoffset = unpack('1I', data[2060:2064])
	samplesize = unpack('1I', data[2064:2068])
	nonsampleoffset = np.fromstring(data[2068:2072], dtype=np.uint32)
	nonsamplesize = np.fromstring(data[2072:2076], dtype=np.uint32)
	ifcenterfrequency = np.fromstring(data[2076:2084], dtype=np.double)
	samplerate = unpack('1d', data[2084:2092])
	bandwidth = unpack('1d', data[2092:2100])
	corrected = unpack('1I', data[2100:2104])
	timetype = unpack('1I', data[2104:2108])
	reftime = unpack('7i', data[2108:2136])
	timesamples = unpack('1Q', data[2136:2144])
	timesamplerate = np.fromstring(data[2144:2152], dtype=np.uint64)
	dataformat = {'datatype': datatype, 'frameoffset': frameoffset, 'framesize':framesize,
		'sampleoffset':sampleoffset, 'samplesize': samplesize, 'nonsampleoffset': nonsampleoffset,
		'nonsamplesize': nonsamplesize, 'ifcenterfrequency': ifcenterfrequency,
		'samplerate': samplerate, 'bandwidth': bandwidth, 'corrected': corrected,
		'timetype': timetype, 'reftime': reftime, 'timesamples': timesamples,
		'timesamplerate': timesamplerate}

	# Get Signal Path and Channel Correction data
	adcscale = np.fromstring(data[3072:3080], dtype=np.double)
	pathdelay = np.fromstring(data[3080:3088], dtype=np.double)
	correctiontype = np.fromstring(data[4096:4100], dtype=np.uint32)
	tableentries = np.fromstring(data[4352:4356], dtype=np.uint32)
	freqindex = 4356
	phaseindex = freqindex + 501*4
	ampindex = phaseindex + 501*4
	freqtable = np.fromstring(data[freqindex:(freqindex+tableentries*4)], dtype=np.float32)
	amptable = np.fromstring(data[phaseindex:(phaseindex+tableentries*4)], dtype=np.float32)
	phasetable = np.fromstring(data[ampindex:(ampindex+tableentries*4)], dtype=np.float32)
	channelcorrection = {'adcscale': adcscale, 'pathdelay': pathdelay, 
		'correctiontype':correctiontype, 'tableentries': tableentries, 
		'freqtable': freqtable, 'amptable': amptable, 'phasetable': phasetable}
	
	metadata = {'versioninfo': versioninfo, 'instrumentstate': instrumentstate,
		'dataformat': dataformat, 'channelcorrection': channelcorrection}

	# Depending on the status of 'display,' display metadata, 
	# correction plots, both, or neither
	if display == '3':
		print_metadata(metadata)
		plot_graphs(metadata)
	elif display == '2':
		print('\nChannel correction graphs plotted.')
		plot_graphs(metadata)
	elif display == '1':
		print_metadata(metadata)
		print('\nMetadata parsed and printed.')
	elif display == '0':
		print('\nData parsed.')
	else: 
		print('Invalid choice for \'metadisplay\' variable. Select 0, 1, 2, or 3.')

	return metadata

def get_adc_samples(infile, metadata, footerflag):
	try:
		data = open(infile, 'rb')
	except IOError:
		print('Invalid input file. Check the \'infile\' and try again.')
		quit()
	filesize = os.path.getsize(infile)

	#Filter file type and read the file appropriately
	if '.r3f' in infile:
		numframes = (filesize/metadata['dataformat']['framesize']) - 1
		print('Number of Frames: %d' % numframes)
		data.seek(metadata['dataformat']['frameoffset'])
		adcdata = np.empty(0)
		rawdata = metadata['dataformat']['nonsampleoffset']
		footerdata = metadata['dataformat']['nonsamplesize']
		footer = np.zeros((numframes, footerdata))
		for i in range(0,numframes):
			frame = data.read(rawdata)
			if footerflag == '0':
				data.seek(footerdata)
			else:
				temp_ftr = data.read(footerdata)
				footer[i] = np.fromstring(temp_ftr, dtype=np.uint8, count=footerdata)
				#footer = parse_footer(temp_ftr)
				print(footer[i])
			adcdata = np.append(adcdata,np.fromstring(frame, dtype=np.int16))
			#print('Current Frame: %d' % i)
	elif '.r3a' in infile:
		adcdata = np.fromfile(infile, dtype=np.int16)

	#Scale ADC data
	adcdata = adcdata*metadata['channelcorrection']['adcscale']
	#print(len(adcdata))
	return adcdata

def parse_footer(footer):
	reserved = np.fromstring(footer, dtype=np.uint16, count=3)
	frame_descr = np.fromstring(footer, dtype=np.uint16, count=1)
	frame_id = np.fromstring(footer, dtype=np.uint32, count=1)
	trigger2_idx = np.fromstring(footer, dtype=np.uint16, count=1)
	trigger1_idx = np.fromstring(footer, dtype=np.uint16, count=1)
	time_sync_idx = np.fromstring(footer, dtype=np.uint16, count=1)
	frame_status = np.fromstring(footer, dtype=np.uint16, count=1)
	timestamp = np.fromstring(footer, dtype=np.uint64, count=1)
	footer = {'frame_descr': frame_descr, 'frame_id': frame_id,
	'trigger2_idx': trigger2_idx, 'trigger1_idx': trigger1_idx,
	'time_sync_idx': time_sync_idx, 'frame_status': frame_status,
	'timestamp': timestamp}
	return footer

def ddc(adcdata,metadata):
	#Generate quadrature signals
	IFfreq = metadata['dataformat']['ifcenterfrequency']
	size = len(adcdata)
	sampleperiod = 1.0/metadata['dataformat']['timesamplerate']
	xaxis = np.linspace(0,size*sampleperiod,size)
	LO_I = np.sin(IFfreq*(2*np.pi)*xaxis)
	LO_Q = np.cos(IFfreq*(2*np.pi)*xaxis)

	#Run ADC data through digital down converter
	I = adcdata*LO_I
	Q = adcdata*LO_Q
	nyquist = metadata['dataformat']['timesamplerate']/2
	cutoff = 40e6/nyquist
	IQfilter = signal.firwin(300, cutoff, window=('kaiser', 10))
	I = signal.lfilter(IQfilter, 1.0, I)
	Q = signal.lfilter(IQfilter, 1.0, Q)
	IQ = I + 1j*Q
	IQ = 2*IQ
	print('Digital Down Conversion Complete.')
	return IQ

def IQ_correction(IQ, metadata):
	#amp and phase correction filter NOT FINISHED
	#CorrectionFrameSize
	framesize = metadata['channelcorrection']['tableentries']
	amptable = metadata['channelcorrection']['amptable']
	phasetable = metadata['channelcorrection']['phasetable']
	#Convert magnitude from dB to V and phase from degrees to rad
	amptable = 1/np.sqrt(10**(amptable/(2**15*10)))
	phasetable = phasetable*np.pi/180
	correct_real = amptable*np.cos(np.radians(phasetable))
	correct_imag = amptable*np.sin(np.radians(phasetable))
	correctFD = correct_real + 1j*correct_imag
	corr_frames = len(IQ)/framesize
	IQi = 0
	
	#Apply correction factors to IQ data frame by frame
	for frame in range(0,corr_frames):
		cframe = IQ[IQi:IQi+framesize]
		cframe = fft(cframe)
		cframe = cframe*correctFD
		cframe = ifft(cframe)
		IQ[IQi:IQi+framesize] = cframe
		IQi = IQi+framesize
	print('IQ Correction Applied.')
	return IQ

def save_mat_file(IQ, metadata, outfile):
	InputCenter = metadata['instrumentstate']['centerfrequency']
	XDelta = 1.0/metadata['dataformat']['timesamplerate']
	Y = IQ
	InputZoom = np.uint8(1)
	Span = metadata['dataformat']['bandwidth']
	sio.savemat(outfile, {'InputCenter':InputCenter,'Span':Span,'XDelta':XDelta,
		'Y':Y,'InputZoom':InputZoom}, format='5')
	print('File saved at %s.' % outfile)
		
def main():
	base_directory = 'C:\\SignalVu-PC Files\\'
	infile = raw_input('Enter input file name including extension (.r3f).\n> ')
	outfile = raw_input('Enter output file name including extension(.mat).\n> ')
	infile = base_directory + infile
	outfile = base_directory + outfile
	
	md_instructions = "0=display nothing\n1=display header data\n2=plot correction tables\n3=display 1 and 2\n> "
	metadisplay = raw_input(md_instructions)

	footerflag = raw_input('0=discard footer\n1=save footer\n> ')
	
	if '.r3f' in infile:
		metadata = get_header_data(infile, metadisplay)
		adcdata = get_adc_samples(infile, metadata, footerflag)
	elif '.r3h' in infile:
		headerinfile = infile
		datainfile = infile[:-1] + 'a'
		metadata = get_header_data(headerinfile, metadisplay)
		adcdata = get_adc_samples(datainfile, metadata, footerflag)
		print('\nYou specified a *.r3h file extension.')
		print('I used the file you specified to get the header data.')
		print('I found the associated *.r3a file and read ADC data from it.')
		print('The file I used is located at{0}'.format(datainfile))
	elif '.r3a' in infile:
		datainfile = infile
		headerinfile = infile[:-1] + 'h'
		metadata = get_header_data(headerinfile, metadisplay)
		adcdata = get_adc_samples(datainfile, metadata, footerflag)
		print('\nYou specified a *.r3a file extension.')
		print('I used the file you specified to get the ADC data.')
		print('I found the associated *.r3h file and read header data from it.')
		print('The file I used is located at {0}'.format(headerinfile))
	else:
		print('\nInvalid input file. Check the \'infile\' variable and try again.')
		quit()

	IQ = ddc(adcdata,metadata)
	#IQ = IQ_correction(IQ,metadata)
	save_mat_file(IQ,metadata,outfile)

if __name__ == '__main__':
	main()