# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 10:48:52 2018

@author: B
"""

import cv2
import numpy as np
import os

def zeroone(img):
    img[img<128]=0
    img[img>=128]=1
    return img
     
def labels(img):
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    connectivity = 4  
    stats = cv2.connectedComponentsWithStats(thresh,connectivity, cv2.CV_32S)
    # The second cell is the label matrix
    labels = stats[1] 
    return labels

def numbercc(img):
    return int(np.unique(img).shape[0])-2

for page in os.listdir('GroundTruth'):
    print('start '+page)
    ind=str(page.split('_')[1].split('.')[0])


    gt=cv2.imread('GroundTruth/msample_'+ind+'.png',0)
    rows,cols=gt.shape
    gt=zeroone(gt)
    gt_labels=labels(gt)
    
    org=255-cv2.imread('CurvedData/sample_'+ind+'.bmp',0)
    org=zeroone(org)
    org_labels=labels(org)
    
    rafi=cv2.imread('RafiOutput/rsample_'+ind+'.bmp',0)
    rafi=zeroone(rafi)
    rafi_labels=labels(rafi)
    #cv2.imwrite('rafi_labels.png',rafi_labels)
    
    f = open('precision/precision'+ind,'w')
    f.write('line number,total connectivity in output,total connectivity in gt intersect output,precision\n')
    
    precisions=[]
    for l in (np.unique(gt_labels)[1:]):
        tgt=np.zeros([rows,cols],dtype=np.uint8)
        tgt[gt_labels==l]=1
        #cv2.imwrite('tgt.png',tgt*255)
        f.write(str(l)+',')    
    
    
        rafi_labels_multiply_tgt=rafi_labels*tgt
        #cv2.imwrite('rafi_labels_multiply_tgt.png',rafi_labels_multiply_tgt*20)
        '''
        trafi=np.zeros([rows,cols],dtype=np.uint8)
        for label in np.unique(tgt_multiply_rafi_labels):
            trafi[rafi_labels==label]=label
        cv2.imwrite('trafi.png',trafi*20)
        '''
        sum_number_cc_rafi_labels=0
        sum_number_cc_rafi_gt_labels=0
        #ps=[]
        for label in np.unique(rafi_labels_multiply_tgt)[1:]:
            number_cc_rafi_labels=0
            number_cc_gt_rafi_labels=0
            trafi=np.zeros([rows,cols],dtype=np.uint8)
            trafi[rafi_labels==label]=1
            org_labels_multiply_trafi=org_labels*trafi
            #cv2.imwrite('out.png',org_labels_multiply_trafi)
            number_cc_rafi_labels=numbercc(org_labels_multiply_trafi)
            if number_cc_rafi_labels<0:
                number_cc_rafi_labels=0
            sum_number_cc_rafi_labels=sum_number_cc_rafi_labels+number_cc_rafi_labels
            org_labels_multiply_trafi_gt=org_labels_multiply_trafi*tgt
            #cv2.imwrite('out.png',org_labels_multiply_trafi_gt)
            number_cc_gt_rafi_labels=numbercc(org_labels_multiply_trafi_gt)
            if number_cc_gt_rafi_labels<0:
                number_cc_gt_rafi_labels=0            
            sum_number_cc_rafi_gt_labels=sum_number_cc_rafi_gt_labels+ number_cc_gt_rafi_labels
            #p=number_cc_gt_rafi_labels/(float(number_cc_rafi_labels)+1e06)
            #ps.append(p)
        
        f.write(str(sum_number_cc_rafi_labels)+',')
        f.write(str(sum_number_cc_rafi_gt_labels)+',')
        #aps=np.mean(ps)
        p=sum_number_cc_rafi_gt_labels/(float(sum_number_cc_rafi_labels)+1e-06)
        f.write(str(p)+',\n')
        precisions.append(p)
        
    precision=np.mean(precisions)
    f.write('Average precision= '+str(precision))
    f.close()
