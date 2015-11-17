"""
Script: RSA300 Streamed Data File Parser
Date: 11/2014
Author: Morgan Allison
Description: This script reads in a streamed data file created by the RSA300,
parses out all the metadata, and saves the raw data.

Directions: 
1. Save a streamed data file from your RSA300
2. Put it somewhere on your computer and enter the file path 
into the 'filename' variable near the bottom of the code
3. Select your display options. They're explained where the variables are

Look for the ####################################################### near the bottom.

Features to add: Apply correction to ADC samples, finish ADC to IQ converter, implement *.mat
file saving feature so that the waveform can be loaded into SignalVu-PC.
"""

import os, time
from struct import *
from math import log,ceil
import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt

# This function simply prints out all the metadata contained in the header
#def print_metadata(metadata, versioninfo, instrumentstate, dataformat, channelcorrection):
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
	print('Samples in Frame: %i' % metadata['dataformat']['framesamples'])
	print('Offset to non-sample data: %i' % metadata['dataformat']['nonsampleoffset'])
	print('Size of non-sample data: %i' % metadata['dataformat']['nonframesamples'])
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
def get_header_data(filename, display):
	datafile = open(filename, 'rb', 64*1024*1024)
	data = datafile.read(16384)

	# Get and print File ID and Version Info sections of the header
	fileid = data[:27]
	#endian = unpack('1I',data[512:516])
	endian = np.fromstring(data[512:516], dtype = np.uint32)
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
	#frameoffset = unpack('1I', data[2052:2056])
	frameoffset = np.fromstring(data[2052:2056], dtype = np.uint32)
	#framesize = unpack('1I', data[2056:2060])
	framesize = np.fromstring(data[2056:2060], dtype = np.uint32)
	sampleoffset = unpack('1I', data[2060:2064])
	sampleoffset = np.fromstring(data[2056:2060], dtype = np.uint32)
	#framesamples = unpack('1I', data[2064:2068])
	framesamples = np.fromstring(data[2064:2068], dtype = np.uint32)
	#nonsampleoffset = unpack('1I', data[2068:2072])
	nonsampleoffset = np.fromstring(data[2068:2072], dtype = np.uint32)
	#nonframesamples = unpack('1I', data[2072:2076])
	nonframesamples = np.fromstring(data[2072:2076], dtype = np.uint32)
	ifcenterfrequency = np.fromstring(data[2076:2084], dtype = np.double)
	samplerate = unpack('1d', data[2084:2092])
	bandwidth = unpack('1d', data[2092:2100])
	corrected = unpack('1I', data[2100:2104])
	timetype = unpack('1I', data[2104:2108])
	reftime = unpack('7i', data[2108:2136])
	timesamples = unpack('1Q', data[2136:2144])
	#timesamplerate = unpack('1Q', data[2144:2152])
	timesamplerate = np.fromstring(data[2144:2152], dtype = np.uint64)
	dataformat = {'datatype': datatype, 'frameoffset': frameoffset, 'framesize':framesize,
		'sampleoffset':sampleoffset, 'framesamples': framesamples, 'nonsampleoffset': nonsampleoffset,
		'nonframesamples': nonframesamples, 'ifcenterfrequency': ifcenterfrequency,
		'samplerate': samplerate, 'bandwidth': bandwidth, 'corrected': corrected,
		'timetype': timetype, 'reftime': reftime, 'timesamples': timesamples,
		'timesamplerate': timesamplerate}

	# Get Signal Path and Channel Correction data
	adcscale = np.fromstring(data[3072:3080], dtype = np.double)
	pathdelay = np.fromstring(data[3080:3088], dtype = np.double)
	correctiontype = np.fromstring(data[4096:4100], dtype = np.uint32)
	tableentries = np.fromstring(data[4352:4356], dtype = np.uint32)
	freqindex = 4356
	phaseindex = freqindex + 501*4
	ampindex = phaseindex + 501*4
	freqtable = np.fromstring(data[freqindex:(freqindex+tableentries*4)], dtype = np.float32)
	amptable = np.fromstring(data[phaseindex:(phaseindex+tableentries*4)], dtype = np.float32)
	phasetable = np.fromstring(data[ampindex:(ampindex+tableentries*4)], dtype = np.float32)
	channelcorrection = {'adcscale': adcscale, 'pathdelay': pathdelay, 
		'correctiontype':correctiontype, 'tableentries': tableentries, 
		'freqtable': freqtable, 'amptable': amptable, 'phasetable': phasetable}
	
	metadata = {'versioninfo': versioninfo, 'instrumentstate': instrumentstate,
		'dataformat': dataformat, 'channelcorrection': channelcorrection}

	# Depending on the status of 'display,' display metadata, 
	# correction plots, both, or neither
	if display == 3:
		print_metadata(metadata)
		plot_graphs(metadata)
	elif display == 2:
		print('\nChannel correction graphs plotted.')
		plot_graphs(metadata)
	elif display == 1:
		print_metadata(metadata)
		print('\nMetadata parsed and printed.')
	elif display == 0:
		print('\nData parsed.')
	else: 
		print('Invalid choice for \'metadisplay\' variable. Select 0, 1, 2, or 3.')

	return metadata
"""#######################################################################################"""
def get_adc_samples_1(filename, metadata):
	t0 = time.clock()
	data = open(filename, 'rb')
	filesize = os.path.getsize(filename)
	print(filesize)


	if '.r3f' in filename:
		numframes = (filesize/metadata['dataformat']['framesize']) - 1
		print('Number of Frames: %d' % numframes)
		data.seek(metadata['dataformat']['frameoffset'])
		adcdata = np.array([numframes*metadata['dataformat']['framesize']])
		rawdata = metadata['dataformat']['nonsampleoffset']
		footerdata = metadata['dataformat']['nonframesamples']
		framesamples = metadata['dataformat']['framesamples']
		for i in range(0,numframes):
			frame = data.read(rawdata)
			data.seek(footerdata,1)
			index1 = framesamples*i
			print(index1)
			index2 = index1 + framesamples
			print(index2)
			intermediate = np.fromstring(frame, dtype = np.int16)
			adcdata[index1:index2] = intermediate
			#adcdata = np.append(adcdata,np.fromstring(frame, dtype = np.int16))
	elif '.r3a' in filename:
		adcdata = np.fromfile(filename, dtype = np.int16)
	
	adcdata = adcdata*metadata['channelcorrection']['adcscale']
	t1 = time.clock()
	print('Time to read and strip data is %f' % (t1-t0))
	return adcdata

"""def get_adc_samples_2(filename, metadata):
	t0 = time.clock()
	data = np.fromfile(filename, dtype = np.int16)
	filesize = len(data)
	print('file size: %d' % filesize)
	framesize = metadata['dataformat']['framesamples']
	numframes = filesize/8192-1
	footersize = 14

	#This mess is supposed to speed up the process of stripping out the footer data
	removeindex = metadata['dataformat']['frameoffset']/2 + metadata['dataformat']['framesamples']
	rai = 0
	removearray = np.zeros([footersize*numframes])
	for i in range(0,numframes):
		for j in range(0,footersize):
			removearray[rai] = removeindex + j
			rai = rai + 1
		removeindex = removeindex + framesize + footersize

	#print(filesize)
	print('Number of frames: %d' % numframes)
	print('Removearray')
	print(np.shape(removearray))
	print(removearray[-10:-1])
	adcdata = np.delete(data, removearray)
	print('The difference between size of data and adcdata should be 43456. It is %d' %(len(data)-len(adcdata)))"""
	
	

def iq_converter(adcdata,metadata,matfilename):
	xlimit = 300
	IFfreq = metadata['dataformat']['ifcenterfrequency']
	#print('IF Frequency: %d' % IFfreq)
	size = len(adcdata)
	#print('ADC Data length: %d' % size)
	samplerate = metadata['dataformat']['timesamplerate']
	sampleperiod = 1.0/samplerate
	t0 = time.clock()
	xaxis = np.linspace(0,size*sampleperiod,size)
	print('Time to create xaxis: %f sec' % (time.clock()-t0))
	t0 = time.clock()
	LO_I = np.sin(IFfreq*2*np.pi*xaxis)
	LO_Q = np.cos(IFfreq*2*np.pi*xaxis)
	print('Time to create LOs: %f sec' % (time.clock()-t0))


	# Running ADC data through quadrature mixer/downconverter
	I = adcdata*LO_I
	Q = adcdata*LO_Q
	nyquist = metadata['dataformat']['timesamplerate']/2
	cutoff = metadata['dataformat']['bandwidth']/nyquist
	to = time.clock()
	DDCfilter = signal.firwin(300, cutoff, window=('kaiser', 10))
	print('Time to create DDC filter: %f sec' % (time.clock()-t0))

	#DDC filter
	t0 = time.clock()
	I = signal.lfilter(DDCfilter, 1.0, I)
	Q = signal.lfilter(DDCfilter, 1.0, Q)
	print('Time to apply filters: %f sec' % (time.clock()-t0))

	plt.figure(3)
	plt.subplot(3,1,1)
	plt.plot(I)
	plt.xlim([0,xlimit])
	plt.subplot(3,1,2)
	plt.plot(Q)
	plt.xlim([0,xlimit])	
	
	#create IQ vector and remove invalid data
	IQ = I + 1j*Q
	IQ = 2*IQ
	IQ = IQ[200:-1]
	
	"""
	#amp and phase correction filter
	#CorrectionFrameSize
	cfs = metadata['channelcorrection']['tableentries']
	amptable = metadata['channelcorrection']['amptable']
	phasetable = metadata['channelcorrection']['phasetable']
	amptable = 1/np.sqrt(10**(amptable/(2**15*10)))
	phasetable = phasetable/2**15*np.pi/180
	correct_real = amptable*np.cos(np.radians(phasetable))
	correct_imag = amptable*np.sin(np.radians(phasetable))
	correctFD = correct_real + 1j*correct_imag
	corr_frames = size/cfs
	IQi = 0
	for frame in range(0,corr_frames):
		cframe = IQ[IQi:IQi+cfs]
		#print(np.shape(cframe))
		cframe = fft(cframe)
		#print(np.shape(cframe))
		cframe = cframe*correctFD
		#print(np.shape(cframe))
		cframe = ifft(cframe)
		#print(np.shape(cframe))4
		IQ[IQi:IQi+cfs] = cframe
		IQi = IQi+cfs
	"""
	"""
	correctTD = ifft(correctFD)
	correctTDext = np.zeros(2*cfs)
	correctTDext = correctTDext+correctTDext*1j
	correctTDext[0:cfs/2-1] = correctTD[0:cfs/2-1]
	correctTDext[3*cfs/2-1:2*cfs] = correctTD[cfs/2-1:cfs]
	correctFD = fft(correctTDext)
	plt.subplot(3,1,1)
	plt.plot(abs(correctTDext))
	plt.subplot(3,1,2)
	plt.plot(abs(correctTD))
	plt.subplot(3,1,3)
	plt.plot(abs(correctFD))
	#plt.show()
	
	frameLength    = 401
	frameLength0_5 = 0.5 * frameLength
	frameLength1_5 = 1.5 * frameLength
	frameLength2   = 2.0 * frameLength
	i1 = 0
	i2 = frameLength

	iqDataCorrected = np.zeros(corr_frames*frameLength,dtype=np.singlecomplex)
	for iFrame in range(0,corr_frames):
		#Select the proper 2048 samples to apply correction to
		if iFrame == 0:
			iqFrame = np.zeros(frameLength2,dtype=np.singlecomplex)
			iqFrame[201:frameLength2] = IQ[0:frameLength1_5]
		elif iFrame == corr_frames:
			iqFrame = np.zeros(frameLength2,dtype=np.singlecomplex)
			iqFrame[0:frameLength1_5] = IQ[-frameLength1_5:-1]
		else:
			iqFrame = IQ[i1-frameLength0_5:i2+frameLength0_5]

		iqFrameFD = fft(iqFrame)
		iqFD = iqFrameFD*correctFD
		iqTD = ifft(iqFD)
		#Extract middle points
		iqDataCorrected[i1:i2] = iqTD[frameLength0_5:frameLength1_5]
		i1 = i1 + frameLength
		i2 = i2 + frameLength
	"""
	
	#freqdomain = np.zeros(size)
	pow2 = ceil(log(size,2))
	nfft = 2**pow2
	IQmag = (np.real(IQ)**2 + np.imag(IQ)**2)/100
	freqdomain = fft(IQmag,nfft)
	spectrum_mag = 10*np.log10(abs(freqdomain*1000)/nfft)
	spectrum_volts = abs(freqdomain)/nfft
	freq = (samplerate/2.0)*np.arange(-1,1,2.0/nfft)
	freq = fftshift(freq)/1e6

	IQmagdB = 10*np.log10(abs(IQmag)*1000)	
	pkpk = np.amax(adcdata) - np.amin(adcdata)
	print('Pk-pk IQ Voltage: %f' % pkpk)
	#print('Max in ToV (dBm): %f' % np.amax(IQmagdB[200:-1]))
	print('Max in ToV (dBm): %f' % np.mean(IQmagdB))
	#print('location of max: %d' % IQmagdB.argmax(axis=0))
	print('Max in Spectrum (dBm): %f' % np.amax(spectrum_mag))
	
	#*.mat saving
	InputCenter = metadata['instrumentstate']['centerfrequency']
	XDelta = sampleperiod
	Y = IQ
	InputZoom = np.uint8(1)
	Span = metadata['dataformat']['bandwidth']

	sio.savemat(matfilename, {'InputCenter':InputCenter,'Span':Span,'XDelta':XDelta,
		'Y':Y,'InputZoom':InputZoom}, format='5')

	"""
	#print(np.shape(freq))
	#print(np.shape(freqdomain))
	plt.figure(1)
	plt.subplot(2,2,1)
	plt.plot(xaxis*1000,np.real(IQ))
	#plt.yticks(np.arange(-.2,.6,.2))
	plt.ylabel('I (V)')
	plt.xlabel('Time (ms)')
	plt.subplot(2,2,2)
	plt.plot(xaxis*1000,np.imag(IQ))
	#plt.yticks(np.arange(-.2,.5,.2))
	plt.ylabel('Q (V)')
	plt.xlabel('Time (ms)')
	#plt.xlim([0, 0.0004])
	plt.subplot(2,2,3)
	plt.plot(xaxis*1000,IQmag)
	#plt.yticks(np.arange(0,.5,.2))
	plt.ylabel('IQ Mag (V)')
	plt.xlabel('Time (ms)')
	#plt.xlim([0, 0.0004])
	plt.subplot(2,2,4)
	plt.plot(xaxis*1000,IQmagdB)
	plt.ylabel('IQ Mag (dBm)')
	#plt.ylim([-100, 0])
	plt.xlabel('Time (ms)')
	#plt.xlim([0, 0.0004])
	"""
	
	plt.subplot(3,1,3)
	plt.plot(IQmagdB)
	plt.xlim([0,xlimit])
	
	#plt.figure(2)
	#plt.subplot(2,1,1)
	#plt.plot(freq,spectrum_volts)
	#plt.xlabel('MHz')
	#plt.xticks(np.arange(freq[0]/1e6,freq[-1]/1e6,freq[-1]/4))
	#plt.xlim([0, 0.0004])
	#plt.plot(freq,spectrum_mag)
	#plt.xlabel('MHz')
	#plt.xlim([-40, 40])
	plt.show()

"""
	amptable = metadata['channelcorrection']['amptable']
	phasetable = metadata['channelcorrection']['phasetable']
	plt.polar(phasetable,amptable)
	plt.show()
"""
	
def main():
	"""################################ ENTER PATH FOR DATA FILE HERE ################################"""
	filename = 'C:\\SignalVu-PC Files\\saved-2015.05.13.15.35.13.254.r3f'
	matfilename = 'C:\\SignalVu-PC Files\\parser\\saved-2015.05.13.15.35.13.254.mat'

	"""################################## ENTER DISPLAY CHOICE HERE ##################################"""
	#metadisplay 0 = don't print anything, 1 = print parsed header data, 
	# 2 = plot correction tables, 3 = print parsed header data and plot correction tables
	metadisplay = 1;

	if '.r3f' in filename:
		metadata = get_header_data(filename, metadisplay)
		adcdata = get_adc_samples_1(filename, metadata)
	elif '.r3h' in filename:
		headerfilename = filename
		datafilename = filename[:-1] + 'a'
		metadata = get_header_data(headerfilename, metadisplay)
		adcdata = get_adc_samples(datafilename, metadata)
		print('\nYou specified a *.r3h file extension.')
		print('I used the file you specified to get the header data.')
		print('I found the associated *.r3a file and read ADC data from it.')
		print('The file I used is located at{0}'.format(datafilename))
	elif '.r3a' in filename:
		datafilename = filename
		headerfilename = filename[:-1] + 'h'
		metadata = get_header_data(headerfilename, metadisplay)
		adcdata = get_adc_samples(datafilename, metadata)
		print('\nYou specified a *.r3a file extension.')
		print('I used the file you specified to get the ADC data.')
		print('I found the associated *.r3h file and read header data from it.')
		print('The file I used is located at {0}'.format(headerfilename))
	else:
		print('Invalid file extension. Check the \'filename\' variable and try again.')
		quit()

	print(filename)
	iq_converter(adcdata,metadata,matfilename)

	#Don't pay attention to this. Srsly. Staaahp.
	"""textfile = open('QPSK2.txt', 'w')
	np.savetxt(textfile, adcdata)
	textfile.close()"""

main()