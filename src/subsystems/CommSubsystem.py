import numpy as np
import math



class CommSubsystem():
    """
        #Implementation of communication subsystem.
        #The data are taken from commercially available components.
    """
    def __init__(self):
        self.POWER = 2 #[W]
        self.RXGAIN = 1 #[dB]
        self.RXLOSS = 0.5 #[dB]
        self.TXGAIN = 1 #[dB]
        self.TXLOSS = 3 #[dB]
        self.FREQUENCY = 437e6 #[Hz]
        self.BANDWIDTH = 9600 #[Hz]
        self.SYMBOLRATE = 9600 #[Hz]
        self.MODULATIONORDER = 4  #QPSK
        self.SENSITIVITY = -151 #[dBW]

    def calculateFreeSpaceLoss(self, dist):
        fsllinear = (4 * np.pi * dist * self.FREQUENCY / 3e8)**2
        return 10 * np.log10(fsllinear)
    
    def calculateNoise(self):
        return 10 * np.log10(1.38e-23 * 290 * self.BANDWIDTH) # kTB assuming temperature of 290K (17C)

    def calculateSNR(self, dist):
        snr = (10 * np.log10(self.POWER) + self.RXGAIN + self.TXGAIN - self.RXLOSS - self.TXLOSS - self.calculateFreeSpaceLoss(dist)) - self.calculateNoise()
        # print(f"Calculated SNR for distance {dist/1000} km: {snr:.2f} dB")
        return snr
    
    def calculateIdealDataRate(self, dist):
        ideal_data_rate = self.BANDWIDTH * np.log2(1 + 10**(self.calculateSNR(dist) / 10)) # bits/s
        # print(f"Calculated Ideal Data Rate for distance {dist/1000} km: {ideal_data_rate:.2f} bits/sec")
        
        if ideal_data_rate > self.SYMBOLRATE * np.log2(self.MODULATIONORDER): # If the channel limits my data rate
            return self.SYMBOLRATE * np.log2(self.MODULATIONORDER)
        else:
            return ideal_data_rate

    def calculateBER(self, dist):
        bitrate = self.SYMBOLRATE * np.log2(self.MODULATIONORDER)
        efficiency = bitrate / self.BANDWIDTH
        ebn0 = 10**(self.calculateSNR(dist) / 10) / efficiency
        ber = (1 / np.log2(self.MODULATIONORDER)) * math.erfc(np.sqrt(2 * ebn0))
        # print(f"Calculated BER for distance {dist/1000} km: {ber:.2e}")
        return ber

    def calculateEffectiveDataRate(self, dist):
        """
            dist: distance in meters
        """
        receivedPower = (10 * np.log10(self.POWER) + self.RXGAIN + self.TXGAIN - self.RXLOSS - self.TXLOSS - self.calculateFreeSpaceLoss(dist))
        # print(f"Calculated received power for distance {dist/1000} km: {receivedPower:.2f} dBW")
        
        if receivedPower >= self.SENSITIVITY:
            ideal_data_rate = self.calculateIdealDataRate(dist)
            ber = self.calculateBER(dist)
            effective_data_rate = ideal_data_rate * (1 - ber) # bits/sec
            # print(f"Calculated Effective Data Rate for distance {dist/1000} km: {effective_data_rate:.2f} bits/sec")
            return effective_data_rate
        else:
            return 0
