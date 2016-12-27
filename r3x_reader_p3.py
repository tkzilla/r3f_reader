"""
Script: RSA300 Streamed Data File Parser
Date: 12/2015
Author: Morgan Allison
Software: Anaconda 2.1.0 (Python 2.7.8, 64-bit) http://continuum.io/downloads
Description: This script reads in a .r3f/.r3a/.r3h file created by the RSA306,
parses out all the metadata, saves the raw data, converts to IQ data, and
exports a .mat file that is readable by SignalVu-PC and a .txt file
containing footer data

Directions: 
1. Save a streamed data file from your RSA300
2. Put it in C:\SignalVu-PC Files\
3. Run the script, enter the name of your source file
4. Choose display features
5. Choose to save or discard IQ data
6. Choose to save or discard footer (iff .r3f file)
-`-----
"""

from struct import unpack
import numpy as np
from scipy.signal import firwin, lfilter
from scipy.io import savemat
from time import perf_counter
from os.path import getsize
import matplotlib.pyplot as plt
import csv

"""##########################Classes##########################"""

class VersionInfo:
	def __init__(self):
		self.fileid = []
		self.endian = []
		self.fFormatVer = []
		self.apiVersion = []
		self.fx3Version = []
		self.fpgaVersion = []
		self.deviceSN = []

class InstrumentState:
	def __init__(self):
		self.refLevel = []
		self.cf = []
		self.temp = []
		self.alignment = []
		self.freqRef = []
		self.trigMode = []
		self.trigSource = []
		self.trigTrans = []
		self.trigLevel = []

class DataFormat:
	def __init__(self):
		self.dataType = []
		self.frameOffset = []
		self.frameSize = []
		self.sampleOffset = []
		self.sampleSize = []
		self.nonsampleOffset = []
		self.nonSampleSize = []
		self.ifcf = []
		self.sRate = []
		self.bandwidth = []
		self.corrected = []
		self.timeType = []
		self.refTime = []
		self.clockSamples = []
		self.timeSampleRate = []
		self.utcTime = []

class ChannelCorrection:
	def __init__(self):
		self.adcScale = []
		self.pathDelay = []
		self.corrType = []
		self.tableEntries = []
		self.freqTable = []
		self.ampTable = []
		self.phaseTable = []

class FooterClass:
	def __init__(self):
		self.frameDescr = []
		self.frameId = []
		self.trig2Idx = []
		self.trig1Idx = []
		self.timeSyncIdx = []
		self.frameStatus = []
		self.timestamp = []

class R3F:
	def __init__(self):
		baseDirectory = 'C:\\SignalVu-PC Files\\'
		fName = input('Enter input file name including extension (.r3f/.r3a/.r3h).\n> ')
		self.inFileName = baseDirectory + fName
		self.outFile = baseDirectory + fName[:-4]
		self.dispFlag = self.headerFlag = self.iqFlag = self.footerFlag = -1024
		dispFlagOptions = ['0', '1', '2', '3']
		otherFlagOptions = ['0', '1']

		while self.dispFlag not in dispFlagOptions:
			self.dispFlag = input("0=display nothing\n1=display header data" +
						"\n2=plot correction tables\n3=display 1 and 2\n> ")
		
		while self.headerFlag not in otherFlagOptions:
			self.headerFlag = input("0=discard header\n1=save header in .csv file\n>")

		while self.iqFlag not in otherFlagOptions:
			self.iqFlag = input("0=discard IQ\n1=save IQ in .mat file\n>")

		if '.r3h' in self.inFileName or '.r3a' in self.inFileName:
			print('\nBecause a .r3f file was not chosen, ' +
				'footer data cannot be extracted.\n')
			self.footerFlag = '0'
		else:
			while self.footerFlag not in otherFlagOptions:
				self.footerFlag = input('0=discard footer\n1=save footer\n> ')

		self.ADC = []
		self.IQ = []
		self.footer = []
		self.vInfo = VersionInfo()
		self.instState = InstrumentState()
		self.dFormat = DataFormat()
		self.chCorr = ChannelCorrection()

	def convert(self):
		# Main conversion function
		# The order of these calls is important because display_control() 
		# and file_muncher() are dependent on data from file_manager()
		self.file_manager()
		self.display_control()
		self.file_muncher()

	def display_control(self):
		# Depending on the status of 'dispFlag,' print header data, 
		# display correction plots, do both, or do neither
		if self.dispFlag == '3':
			self.print_header_data()
			print('Header printed and channel correction graphs plotted.\n')
			self.plot_graphs()
		elif self.dispFlag == '2':
			print('Channel correction graphs plotted.\n')
			self.plot_graphs()
		elif self.dispFlag == '1':
			self.print_header_data()
			print('Header printed.\n')
		elif self.dispFlag== '0':
			print('Header parsed.')
		else: 
			print('Invalid choice. Select 0, 1, 2, or 3.')
			quit()

	def get_header_data(self):
		# Parses the header section of *.r3f and *.r3h files.
		# Certain fields use np.fromstring() rather than unpack() because
		# np.fromstring() allows the user to specify data type.
		# unpack() saves data as a tuple, which can be used for printing
		# but not calculations
		data = self.headerFile.read(16384)

		# Get File ID and Version Info sections of the header
		self.vInfo.fileid = data[:27]
		self.vInfo.endian = np.fromstring(
			data[512:516], dtype=np.uint32)
		self.vInfo.fFormatVer = unpack('4B', data[516:520])
		self.vInfo.apiVersion = unpack('4B', data[520:524])
		self.vInfo.fx3Version = unpack('4B', data[524:528])
		self.vInfo.fpgaVersion = unpack('4B', data[528:532])
		self.vInfo.deviceSN = data[532:596]
		
		# Get the Instrument State section of the header
		self.instState.refLevel = unpack('1d', data[1024:1032])
		self.instState.cf = unpack('1d', data[1032:1040])
		self.instState.temp = unpack('1d', data[1040:1048])
		self.instState.alignment = unpack('1I', data[1048:1052])
		self.instState.freqRef = unpack('1I', data[1052:1056])
		self.instState.trigMode = unpack('1I', data[1056:1060])
		self.instState.trigSource = unpack('1I', data[1060:1064])
		self.instState.trigTrans = unpack('1I', data[1064:1068])
		self.instState.trigLevel = unpack('1d', data[1068:1076])

		# Get Data Format section of the header
		#self.dFormat.dataType = unpack('1I', data[2048:2052])
		self.dFormat.dataType = np.fromstring(
			data[2048:2052], dtype=np.uint32)
		if self.dFormat.dataType == 161:
			self.dFormat.dataType = 2 #bytes per sample
		self.dFormat.frameOffset = np.fromstring(
			data[2052:2056], dtype=np.uint32)
		self.dFormat.frameSize = np.fromstring(
			data[2056:2060], dtype=np.uint32)
		self.dFormat.sampleOffset = unpack('1I', data[2060:2064])
		self.dFormat.sampleSize = np.fromstring(
			data[2064:2068], dtype=np.int32)
		self.dFormat.nonsampleOffset = np.fromstring(
			data[2068:2072], dtype=np.uint32)
		self.dFormat.nonSampleSize = np.fromstring(
			data[2072:2076], dtype=np.uint32)
		self.dFormat.ifcf = np.fromstring(
			data[2076:2084], dtype=np.double)
		#self.dFormat.sRate = unpack('1d', data[2084:2092])
		self.dFormat.sRate = np.fromstring(
			data[2084:2092], dtype=np.double)
		self.dFormat.bandwidth = unpack('1d', data[2092:2100])
		self.dFormat.corrected = unpack('1I', data[2100:2104])
		self.dFormat.timeType = unpack('1I', data[2104:2108])
		self.dFormat.refTime = unpack('7i', data[2108:2136])
		self.dFormat.clockSamples = unpack('1Q', data[2136:2144])
		self.dFormat.timeSampleRate = np.fromstring(
			data[2144:2152], dtype=np.uint64)
		self.dFormat.utcTime = unpack('7i', data[2152:2180])

		# Get Signal Path and Channel Correction data
		self.chCorr.adcScale = np.fromstring(
			data[3072:3080], dtype=np.double)
		self.chCorr.pathDelay = np.fromstring(
			data[3080:3088], dtype=np.double)
		self.chCorr.corrType = np.fromstring(
			data[4096:4100], dtype=np.uint32)
		tableEntries = np.fromstring(data[4352:4356], dtype=np.uint32)
		self.chCorr.tableEntries = tableEntries	#purely for use in IQ_correction()
		freqindex = 4356
		phaseindex = freqindex + 501*4
		ampindex = phaseindex + 501*4
		self.chCorr.freqTable = np.fromstring(
			data[freqindex:(freqindex+501*4)], dtype=np.float32)
		self.chCorr.phaseTable = np.fromstring(
			data[phaseindex:ampindex], dtype=np.float32)
		self.chCorr.ampTable = np.fromstring(
			data[ampindex:(int(ampindex+tableEntries*4))], dtype=np.float32)
		self.headerFile.close()
		
	def print_header_data(self):
		# This function simply prints out all the header data
		print('\nFILE INFO')
		print('FileID: ', self.vInfo.fileid.decode())
		print('Endian Check: {:#x}'.format(self.vInfo.endian[0]))
		print('File Format Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
			self.vInfo.fFormatVer))
		print('API Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
			self.vInfo.apiVersion))
		print('FX3 Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
			self.vInfo.fx3Version))
		print('FPGA Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
			self.vInfo.fpgaVersion))
		print('Device Serial Number: ', self.vInfo.deviceSN.decode())
		# De facto \n in device serial number

		print('INSTRUMENT STATE')
		print('Reference Level: {:.2f} dBm'.format(
			self.instState.refLevel[0]))
		print('Center Frequency: {} Hz'.format(
			self.instState.cf[0]))
		print('temp: {} C'.format(self.instState.temp[0]))
		print('Alignment status: ', self.instState.alignment[0])
		print('Frequency Reference: ', self.instState.freqRef[0])
		print('Trigger mode: ', self.instState.trigMode[0])
		print('Trigger Source: ', self.instState.trigSource[0])
		print('Trigger Transition: ', self.instState.trigTrans[0])
		print('Trigger Level: {} dBm'.format(self.instState.trigLevel[0]))

		print('\nDATA FORMAT')
		print('Data Type: {} bytes per sample'.format(self.dFormat.dataType))
		print('Offset to first frame (bytes): ', self.dFormat.frameOffset[0])
		print('Frame Size (bytes): ', self.dFormat.frameSize[0])
		print('Offset to sample data (bytes): ', self.dFormat.sampleOffset[0])
		print('Samples in Frame: ', self.dFormat.sampleSize[0])
		print('Offset to non-sample data: ', self.dFormat.nonsampleOffset[0])
		print('Size of non-sample data: ', self.dFormat.nonSampleSize[0])
		print('IF Center Frequency: {} Hz'.format(
			self.dFormat.ifcf[0]))
		print('Sample Rate: {} S/sec'.format(self.dFormat.sRate[0]))
		print('Bandwidth: {} Hz'.format(self.dFormat.bandwidth[0]))
		print('Corrected data status: ', self.dFormat.corrected[0])
		print('Time Type (0=local, 1=remote): ', self.dFormat.timeType[0])
		print('Reference Time: {0[1]}/{0[2]}/{0[0]} at {0[3]}:{0[4]}:{0[5]}.{0[6]}'.format(self.dFormat.refTime))
		print('Clock sample count: ', self.dFormat.clockSamples[0])
		print('Sample ticks per second: ', self.dFormat.timeSampleRate[0])
		print('UTC Time: {0[1]}/{0[2]}/{0[0]} at {0[3]}:{0[4]}:{0[5]}.{0[6]}'.format(self.dFormat.utcTime))

		print('\nCHANNEL AND SIGNAL PATH CORRECTION')
		print('ADC Scale Factor: ', self.chCorr.adcScale[0])
		print('Signal Path Delay: {} nsec'.format(self.chCorr.pathDelay[0]*1e9))
		print('Correction Type (0=LF, 1=IF): ', self.chCorr.corrType[0])


	def plot_graphs(self):
		# This function plots the amplitude and phase correction 
		# tables as a function of IF frequency
		plt.subplot(2,1,1)
		plt.plot(self.chCorr.freqTable/1e6,self.chCorr.ampTable)
		plt.title('Amplitude and Phase Correction')
		plt.ylabel('Amplitude (dB)')
		plt.subplot(2,1,2)
		plt.plot(self.chCorr.freqTable/1e6,self.chCorr.phaseTable)
		plt.ylabel('Phase (degrees)')
		plt.xlabel('IF Frequency (MHz)')
		plt.show()
		plt.clf()

	def file_manager(self):
		# Creates header and data files out of user's chosen file for use in
		# other functions, calculates file size in bytes and length in
		# seconds, and calls get_header_data()
		if '.r3f' in self.inFileName:
			try:
				self.headerFile = open(self.inFileName, 'rb')
				self.dataFile = open(self.inFileName, 'rb')
			except IOError:
				print('\nCannot open file. Check the input file name and try again.\n')
				quit()
			self.get_header_data()
			self.fileSize = getsize(self.dataFile.name)
			self.numFrames = int((self.fileSize/self.dFormat.frameSize) - 1)
			#because it needs to be unsigned to check looping in file_muncher correctly
			self.fileLength = self.numFrames*(self.dFormat.sampleSize/self.dFormat.sRate)
			print('Number of Frames: {}\n'.format(self.numFrames))
			print('File size is: {} bytes.'.format(self.fileSize))
			print('File length is: {} seconds.'.format(self.fileLength))

		elif '.r3a' in self.inFileName or '.r3h' in self.inFileName:
			try:	
				self.headerFile = open((self.inFileName[:-1] + 'h'), 'rb')
				self.dataFile = open((self.inFileName[:-1] + 'a'), 'rb')
			except IOError:
				print('\nCannot open file. Check the input file name and try again.\n')
				quit()
			self.get_header_data()
			self.fileSize = int(getsize(self.dataFile.name))
			self.fileLength = self.fileSize/self.dFormat.dataType/self.dFormat.sRate
			print('File size is: {} bytes.\nFile Length is {} seconds.\n'.format(
				self.fileSize, self.fileLength))
		else:
			print('Compatible file extension not found, check input file and try again.')
			quit()

	def file_muncher(self):
		# Determines if the file is > 1 second long and splits accordingly
		# and calls get_adc_samples(), ddc(), and file_saver(),
		# which are really the core processing components of this script
		print('Beginning file conversion.\n')
		loop = 0
		# formatted processing loop based on number of frames in data file
		if '.r3f' in self.dataFile.name:
			fps = 13698
			while self.numFrames > 0:
				if self.numFrames > fps:
					print('File is longer than 1 second, splitting into {} files.'
						.format(int(np.ceil(self.fileLength))))
					processData = fps
				else:
					processData = self.numFrames
				if loop == 0:
					startPoint = self.dFormat.frameOffset
				elif loop > 0:
					startPoint = loop*fps*self.dFormat.frameSize
				self.get_adc_samples(processData, startPoint)
				self.ddc()
				self.file_saver(loop, processData)
				self.numFrames -= fps
				loop += 1
		# raw processing loop based on file size in bytes
		elif '.r3a' in self.dataFile.name:
			bytesPerSecond = 224000000
			while self.fileSize > 0:
				if self.fileSize > bytesPerSecond:
					processData = bytesPerSecond/2 #two bytes per sample
				else:
					processData = self.fileSize
				if loop == 0:
					startPoint = 0
				elif loop > 0:
					startPoint = loop*bytesPerSecond
				self.get_adc_samples(processData, startPoint)
				self.ddc()
				self.file_saver(loop, processData)
				self.fileSize -= bytesPerSecond
				loop += 1
		self.dataFile.close()

	def get_adc_samples(self, processData, startPoint):
		# Reads ADC samples from input file 
		# Skips over or saves footer for .r3f files
		# Reads in everything for .r3a files

		#Filter file type and read the file appropriately
		if '.r3f' in self.dataFile.name:
			self.dataFile.seek(startPoint[0])
			adcsamples = np.empty(processData*self.dFormat.sampleSize[0])
			fstart = 0
			fstop = self.dFormat.sampleSize[0]
			# self.footer = np.arange(processData)
			self.footer = []
			for i in range(processData):
				frame = self.dataFile.read(self.dFormat.nonsampleOffset[0])
				adcsamples[fstart:fstop] = np.fromstring(frame, dtype=np.int16)
				fstart = fstop
				fstop = fstop + self.dFormat.sampleSize[0]
				if self.footerFlag == '0':
					self.dataFile.seek(self.dFormat.nonSampleSize[0],1)
				else:
					temp_ftr = self.dataFile.read(self.dFormat.nonSampleSize[0])
					# self.footer[i] = FooterClass()
					self.footer.append(self.parse_footer(temp_ftr))
		elif '.r3a' in self.dataFile.name:
			self.dataFile.seek(startPoint)
			adcsamples = np.empty(processData)
			adcsamples = np.fromfile(self.dataFile, dtype=np.int16, 
				count=processData)
		else:
			print('Invalid file type. Please specify a .r3f, r3a, or .r3h file.\n')
			quit()

		#Scale ADC data
		self.ADC = adcsamples*self.chCorr.adcScale

	def parse_footer(self, raw_footer):
		# Parses footer based on internal footer documentation
		footer = FooterClass()
		footer.reserved = np.fromstring(raw_footer[0:6], 
			dtype=np.uint16, count=3)
		footer.frameId = np.fromstring(raw_footer[8:12],
		 dtype=np.uint32, count=1)
		footer.trig2Idx = np.fromstring(raw_footer[12:14],
		 dtype=np.uint16, count=1)
		footer.trig1Idx = np.fromstring(raw_footer[14:16],
		 dtype=np.uint16, count=1)
		footer.timeSyncIdx = np.fromstring(raw_footer[16:18],
		 dtype=np.uint16, count=1)
		footer.frameStatus = '{0:8b}'.format(
			int(np.fromstring(raw_footer[18:20], dtype=np.uint16, count=1)))
		footer.timestamp = np.fromstring(raw_footer[20:28], 
			dtype=np.uint64, count=1)
		
		return footer

	def ddc(self):
		# Digital downconverter that converts ADC data to IQ data
		if self.iqFlag =='1':
			#Generate quadrature signals
			ifFreq = self.dFormat.ifcf
			size = len(self.ADC)
			samplePeriod = 1.0/self.dFormat.timeSampleRate
			xAxis = np.linspace(0,size*samplePeriod,size)
			loI = np.sin(ifFreq*(2*np.pi)*xAxis)
			loQ = np.cos(ifFreq*(2*np.pi)*xAxis)
			del(xAxis)

			#Run ADC data through digital downconverter
			I = self.ADC*loI
			Q = self.ADC*loQ
			del(loI)
			del(loQ)
			nyquist = self.dFormat.timeSampleRate/2
			cutoff = 40e6/nyquist
			IQfilter = firwin(32, cutoff, window=('kaiser', 2.23))
			I = lfilter(IQfilter, 1.0, I)
			Q = lfilter(IQfilter, 1.0, Q)
			IQ = I + 1j*Q
			IQ = 2*IQ
			self.IQ = IQ


	def export_header_data(self):
		# Experimental
		# This function exports header data to a csv file
		fName = self.outFile + '.csv'
		with open(fName, 'w') as csvfile:
			hWriter = csv.writer(csvfile)
			hWriter.writerow('\nFILE INFO')
			hWriter.writerow('FileID: {}'.format(self.vInfo.fileid.decode()))
			hWriter.writerow('Endian Check: {:#x}'.format(self.vInfo.endian[0]))
			hWriter.writerow('File Format Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
				self.vInfo.fFormatVer))
			hWriter.writerow('API Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
				self.vInfo.apiVersion))
			hWriter.writerow('FX3 Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
				self.vInfo.fx3Version))
			hWriter.writerow('FPGA Version: {0[0]}.{0[1]}.{0[2]}.{0[3]}'.format(
				self.vInfo.fpgaVersion))
			hWriter.writerow('Device Serial Number: {}'.format(self.vInfo.deviceSN.decode()))

			hWriter.writerow('\nINSTRUMENT STATE')
			hWriter.writerow('Reference Level: {:.2f} dBm'.format(
				self.instState.refLevel[0]))
			hWriter.writerow('Center Frequency: {} Hz'.format(
				self.instState.cf[0]))
			hWriter.writerow('temp: {} C'.format(self.instState.temp[0]))
			hWriter.writerow('Alignment status: {}'.format(self.instState.alignment[0]))
			hWriter.writerow('Frequency Reference: {}'.format(self.instState.freqRef[0]))
			hWriter.writerow('Trigger mode: {}'.format(self.instState.trigMode[0]))
			hWriter.writerow('Trigger Source: {}'.format(self.instState.trigSource[0]))
			hWriter.writerow('Trigger Transition: {}'.format(self.instState.trigTrans[0]))
			hWriter.writerow('Trigger Level: {} dBm'.format(self.instState.trigLevel[0]))

			hWriter.writerow('\nDATA FORMAT')
			hWriter.writerow('Data Type: {} bytes per sample'.format(self.dFormat.dataType))
			hWriter.writerow('Offset to first frame (bytes): {}'.format(self.dFormat.frameOffset[0]))
			hWriter.writerow('Frame Size (bytes): {}'.format(self.dFormat.frameSize[0]))
			hWriter.writerow('Offset to sample data (bytes): {}'.format(self.dFormat.sampleOffset[0]))
			hWriter.writerow('Samples in Frame: {}'.format(self.dFormat.sampleSize[0]))
			hWriter.writerow('Offset to non-sample data: {}'.format(self.dFormat.nonsampleOffset[0]))
			hWriter.writerow('Size of non-sample data: {}'.format(self.dFormat.nonSampleSize[0]))
			hWriter.writerow('IF Center Frequency: {} Hz'.format(
				self.dFormat.ifcf[0]))
			hWriter.writerow('Sample Rate: {} S/sec'.format(self.dFormat.sRate[0]))
			hWriter.writerow('Bandwidth: {} Hz'.format(self.dFormat.bandwidth[0]))
			hWriter.writerow('Corrected data status: {}'.format(self.dFormat.corrected[0]))
			hWriter.writerow('Time Type (0=local, 1=remote): {}'.format(self.dFormat.timeType[0]))
			hWriter.writerow('Reference Time: {0[1]}/{0[2]}/{0[0]} at {0[3]}:{0[4]}:{0[5]}.{0[6]}'.format(self.dFormat.refTime))
			hWriter.writerow('Clock sample count: {}'.format(self.dFormat.clockSamples[0]))
			hWriter.writerow('Sample ticks per second: {}'.format(self.dFormat.timeSampleRate[0]))
			hWriter.writerow('UTC Time: {0[1]}/{0[2]}/{0[0]} at {0[3]}:{0[4]}:{0[5]}.{0[6]}\n'.format(self.dFormat.utcTime))

			hWriter.writerow('\nCHANNEL AND SIGNAL PATH CORRECTION')
			hWriter.writerow('ADC Scale Factor: {}'.format(self.chCorr.adcScale[0]))
			hWriter.writerow('Signal Path Delay: {} nsec'.format(self.chCorr.pathDelay[0]*1e9))
			hWriter.writerow('Correction Type (0=LF, 1=IF): {}'.format(self.chCorr.corrType[0]))

					
	def file_saver(self, loop, processData):
		# Saves a .mat file containing variables specified in the 
		# SignalVu-PC help file
		# Also saves a footer data in a .txt file
		if self.headerFlag == '1':
			self.export_header_data()
		
		if self.iqFlag == '1':
			InputCenter = self.instState.cf
			XDelta = 1.0/self.dFormat.timeSampleRate
			Y = self.IQ
			InputZoom = np.uint8(1)
			Span = self.dFormat.bandwidth
			fileName = self.outFile + '_' + str(loop)
			savemat(fileName, {'InputCenter':InputCenter,'Span':Span, 
				'XDelta':XDelta,'Y':Y,'InputZoom':InputZoom}, format='5')
			print('Data file saved at {}.mat'.format(fileName))

		if self.footerFlag == '1':
			fName = self.outFile + '_' + str(loop) + '.txt'
			ffile = open(fName, 'w')
			ffile.write('FrameID\tTrig1\tTrig2\tTSync\tFrmStatus\tTimeStamp\n')
			for i in range(processData):
				ffile.write(', '.join(map(str, self.footer[i].frameId)))
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].trig2Idx)))
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].trig1Idx)))
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].timeSyncIdx)))
				ffile.write('\t')
				ffile.write(self.footer[i].frameStatus)
				ffile.write('\t')
				ffile.write(', '.join(map(str, self.footer[i].timestamp)))
				ffile.write('\n')
			ffile.close()
			print('Footer file saved at {}.'.format(fName))

def main():
	r3f = R3F()
	start = perf_counter()
	r3f.convert()
	end = perf_counter()
	print('Elapsed time is: {}'.format((end-start)))

if __name__ == '__main__':
	main()
