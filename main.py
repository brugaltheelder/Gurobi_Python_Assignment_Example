__author__ = 'troy'
from scipy import io
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
from time import time
from gurobipy import *


class dao_data(object):
    def __init__(self,inputFilename,bound):
        matFile = io.loadmat(inputFilename)
        self.nVox,self.beams,self.dataFolder = int(matFile['nVox']), np.array(matFile['beams'].flatten()),str(matFile['dataFolder'][0])
        self.nBeams,self.nBPB,self.nDIJSPB = len(self.beams),np.array(matFile['nBPB'].flatten()),np.array(matFile['nDIJSPB'].flatten())
        self.aOver, self.aUnder, self.thresh = np.array(matFile['aOver'].flatten()),np.array(matFile['aUnder'].flatten()),np.array(matFile['thresh'].flatten())
        self.currentDose, self.objDescent,self.aperData, self.rowStart = np.array([]),[],[],[matFile['rowStart'][0][:][b][0] for b in range(self.nBeams)]
        self.aperBound, self.stageBound = bound,[]
        self.ongoingDose = {}

        #Read in all D matrices
        self.Dij =  []
        print 'loading Dij files into memory...'
        for b in range(self.nBeams):
            Dij = np.fromfile(self.dataFolder +'bixel' + str(self.beams[b]) + '.bin',dtype = np.float32).reshape((self.nDIJSPB[b],3))
            self.Dij.append(sps.csr_matrix((Dij[:][:,2],(Dij[:][:,0]-1,Dij[:][:,1]-1)),shape=(self.nBPB[b],self.nVox)))

    def printAllData(self):
        #Prints data
        print self.nVox,self.beams,self.dataFolder,self.nBeams,self.nBPB,self.nDIJSPB,self.aOver,self.aUnder,self.thresh


class dao_model(object):
    def __init__(self, data):
        #Initialized variables in restricted master problem
        self.y,self.m = [],Model('DAO_RMP')
        print 'Adding Variables...',
        self.z = [self.m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS) for i in xrange(data.nVox)]
        self.under = [self.m.addVar(vtype=GRB.CONTINUOUS,lb=0) for i in xrange(data.nVox)]
        self.over = [self.m.addVar(lb=0.0,vtype=GRB.CONTINUOUS) for i in xrange(data.nVox)]
        self.m.update()
        print 'Adding Constraints...',
        self.doseConstr = [self.m.addConstr(-self.z[j],GRB.EQUAL,0) for j in xrange(data.nVox)]
        self.underConstr = [self.m.addConstr(self.under[j],GRB.GREATER_EQUAL,data.thresh[j]-self.z[j]) for j in xrange(data.nVox)]
        self.overConstr = [self.m.addConstr(self.over[j],GRB.GREATER_EQUAL,self.z[j]-data.thresh[j]) for j in xrange(data.nVox)]
        #add constraint for each beam dose
        self.aperBound = [self.m.addConstr(0.0,GRB.LESS_EQUAL,data.aperBound) for b in xrange(data.nBeams)]
        self.m.update()
        print 'Building Objective...',
        self.obj = QuadExpr()
        self.obj.addTerms(data.aOver,self.over,self.over)
        self.obj.addTerms(data.aUnder,self.under,self.under)
        self.m.setObjective(self.obj)
        print 'Model Initialized'
    def solveAndUpdate(self,data):
        #optimizes model and saves dose distribution

        self.m.optimize()
        data.objDescent.append(self.m.ObjVal)
        data.currentDose = np.array([self.z[j].X for j in xrange(data.nVox)])
        data.totalAperDose =[0]*data.nBeams
        for ap in range(len(data.aperData)):
            data.totalAperDose[data.aperData[ap].b]+= self.y[ap].X
        data.ppOffset = [None for i in range(data.nBeams)]
        for i in range(data.nBeams):
            if data.totalAperDose[i]>=data.aperBound-1e-5:
                for ap in range(len(data.aperData)):
                    if data.aperData[ap].b==i:
                        data.ppOffset[i] = data.aperData[ap].beamlets[:]
                        break
        data.ongoingDose['iter'+str(len(self.y))] = np.array([self.z[j].X for j in xrange(data.nVox)])
        io.savemat('ongoingDose.mat',data.ongoingDose)
    def addAper(self,data,b,beamlets):
        #generates an aperture object and adds a variable to the model
        data.aperData.append(dao_aper(data,b,beamlets))
        doseCol = Column(data.aperData[-1].voxelDose.tolist(),self.doseConstr)
        doseCol.addTerms(1,self.aperBound[b])
        self.y.append(self.m.addVar(lb=0.0,vtype = GRB.CONTINUOUS,column = doseCol)); self.m.update()

    def pricingProblemBeamSimple(self,data,b):
        #returns simple pricing problem worth
        gradient= self.getGradient(data,b)
        worth = gradient[gradient<0].sum()
        if data.ppOffset[b]!=None:
            worth-=self.getGradAper(data.ppOffset[b],gradient)
        return worth,np.nonzero(gradient<0)[0]
    def pricingProblemBeamMLC(self,data,b):
        gradient,beamlets,worth = self.getGradient(data,b),[],0

        for l in range(len(data.rowStart[b])-1):
            maxSoFar, maxEndingHere,lE,rE,=0,0,data.rowStart[b][l],data.rowStart[b][l]
            for i in range(data.rowStart[b][l],data.rowStart[b][l+1]):
                maxEndingHere+=gradient[i]
                if maxEndingHere>0:
                    maxEndingHere,lE,rE = 0,i+1,i+1
                if maxSoFar>maxEndingHere:
                    maxSoFar,rE = maxEndingHere,i+1
            for i in range(lE,rE):
                beamlets.append(i)

            worth+=maxSoFar
        if data.ppOffset[b]!=None:
            worth-=self.getGradAper(data.ppOffset[b],gradient)
        return worth,beamlets

    def getGradAper(self,beamlets,gradient):
        return sum([gradient[beamlets[i]] for i in range(len(beamlets))])


    def pricingProblemMLC(self,data):
        #TODO Implement MLC-oriented pricing problem
        bestWorth,bestBeamlets,bestBeam = 0,np.array([]),-1
        stageBound = self.m.objVal
        for b in range(data.nBeams):
            worthHolder,beamletsHolder = self.pricingProblemBeamMLC(data,b)
            stageBound+=data.aperBound*worthHolder
            if worthHolder<bestWorth:
                bestWorth,bestBeamlets,bestBeam = worthHolder,beamletsHolder[:],b
        data.stageBound.append(stageBound)
        return bestBeam,bestBeamlets

    def pricingProblemSimple(self,data):
        #Finds the best beam using the simple pricing problem
        bestWorth,bestBeamlets,bestBeam = 0,np.array([]),-1
        stageBound = self.m.objVal
        for b in range(data.nBeams):
            worthHolder,beamletsHolder = self.pricingProblemBeamSimple(data,b)
            stageBound+=data.aperBound*worthHolder
            if worthHolder<bestWorth:
                bestWorth,bestBeamlets,bestBeam = worthHolder,beamletsHolder[:],b
        data.stageBound.append(stageBound)
        return bestBeam,bestBeamlets
    def getGradient(self,data,b):
        #Generates gradient for a particular beam given a dose distribution
        #Dij = np.fromfile(data.dataFolder +'bixel' + str(data.beams[b]) + '.bin',dtype = np.float32).reshape((data.nDIJSPB[b],3))
        #Dij = np.fromfile(data.dataFolder +'beam' + str(data.beams[b]) + 'bixDs.bin',dtype = np.float32).reshape((data.nDIJSPB[b],3))
        #print len(data.currentDose), len(data.thresh)
        oDose,uDose = data.currentDose-data.thresh, data.currentDose-data.thresh
       # print oDose,uDose
        oDose[oDose<0],uDose[uDose>0]=0,0
        #return 2*sps.csr_matrix((Dij[:][:,2],(Dij[:][:,0]-1,Dij[:][:,1]-1)),shape=(data.nBPB[b],data.nVox)).dot(oDose * data.aOver + uDose * data.aUnder)
        return 2*data.Dij[b].dot(oDose * data.aOver + uDose * data.aUnder)
    def writeSolution(self,data,outName):
        #This saves the apertures to a matlab cell array
        io.savemat(outName,{'solution':[{'b':data.beams[data.aperData[b].b],'beamlets':data.aperData[b].beamlets,'y':self.y[b].X} for b in range(len(data.aperData)) ],'dose':data.currentDose, 'obj':data.objDescent, 'bound':data.stageBound})
    def plotDescentObj(self,data,case,outTag,bound):
        #This plots the descent of the objective function
        plt.semilogy(np.arange(len(data.objDescent)),data.objDescent)
        plt.title('Objective descent for case '+case)
        plt.xlabel('Iteration'); plt.ylabel('Log of objective')
        plt.savefig('obj'+case+'_'+outTag+'_'+str(bound)+'.png',dpi=300)
        plt.close()
    def plotDescentBound(self,data,case,outTag,bound):
        #This plots the descent of the objective function
        plt.plot(np.arange(len(data.stageBound)),data.stageBound)
        plt.title('Objective descent for case '+case)
        plt.xlabel('Iteration'); plt.ylabel('Log of objective')
        plt.savefig('bound'+case+'_'+outTag+'_'+str(bound)+'.png',dpi=300)
        plt.close()

class dao_aper(object):
    '''
    This holds the data necessary to define an aperture. It saves the subset of beamlets in the aperture
    and then calculates the aperture dose.
    '''
    def __init__(self,data,b,beamlets):
        self.b,self.beamlets,self.voxelDose = b,beamlets[:],np.zeros(data.nVox)
        self.populateDose(data)
    def populateDose(self,data):
        #Dij = np.fromfile(data.dataFolder +'bixel' + str(data.beams[self.b]) + '.bin',dtype = np.float32).reshape((-1,3))
        bBool = np.zeros(data.nBPB[self.b])
        bBool[np.ix_(self.beamlets)] = 1
        self.voxelDose = data.Dij[self.b].transpose().dot(bBool)
        #self.voxelDose = sps.csr_matrix((Dij[:][:,2],(Dij[:][:,0]-1,Dij[:][:,1]-1)),shape=(data.nBPB[self.b],data.nVox)).transpose().dot(bBool)

def runDAO(case,aperLimit,outTag):

    startTime = time()
    #Initialize model and data objects
    data = dao_data('pydata'+case+'.mat')
    model = dao_model(data)

    #Generate DAO loop
    while True:
        #Solve RMP and update dose vector
        model.solveAndUpdate(data)
        #Stopping conditions
        if len(model.y)==aperLimit:
            break
        print 'Generating new aperture'
        #Run pricing problem and add best aperture
        bHolder,beamletsHolder = model.pricingProblemSimple(data)
        model.addAper(data,bHolder,beamletsHolder)

    #Output solution and plot the objective descent
    model.writeSolution(data,'out'+case+'_'+outTag+'.mat')
    model.writeSolution(data,'out'+case+'.mat')
    #model.plotDescentObj(data,case,outTag)
    print time()-startTime,' time elapsed for case '+ case + ' over '+ str(aperLimit) + ' iterations'

def runDAO_MLC(case,aperLimit,outTag,filename,bound):

    startTime = time()
    #Initialize model and data objects
    data = dao_data(filename,bound)
    model = dao_model(data)

    #Generate DAO loop
    while True:
        #Solve RMP and update dose vector
        model.solveAndUpdate(data)
        #Stopping conditions
        if len(model.y)==aperLimit:
            break
        print 'Generating new aperture'
        #Run pricing problem and add best aperture
        bHolder,beamletsHolder = model.pricingProblemSimple(data)
        #bHolder,beamletsHolder = model.pricingProblemMLC(data)
        model.addAper(data,bHolder,beamletsHolder)

    #Output solution and plot the objective descent
    model.writeSolution(data,'out'+case+'_'+outTag+'_'+str(aperLimit)+'_'+str(bound)+'.mat')
    #model.plotDescentObj(data,case,outTag,bound)
    #model.plotDescentBound(data,case,outTag,bound)
    print time()-startTime,' time elapsed for case '+ case + ' over '+ str(aperLimit) + ' iterations'


#Parameters
aperLimit, case, filename = 150, 'lungmpc5', 'pydaodatalungmpc5.mat'
bound = 40 # also 40

runDAO_MLC(case,aperLimit,'simple', filename,bound)
#runDAO(case,aperLimit,'simple')

