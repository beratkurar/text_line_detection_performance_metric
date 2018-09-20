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
    
    f = open('recall/recall'+ind,'w')
    f.write('line number,connectivity in gt,connectivity in output,recall\n')
    
    
    recalls=[]
    for l in (np.unique(gt_labels)[1:]):
        tgt=np.zeros([rows,cols],dtype=np.uint8)
        tgt[gt_labels==l]=1   
        #cv2.imwrite('tgt.png',tgt*255)
        f.write(str(l)+',') 
    
        org_labels_multiply_tgt=org_labels*tgt
        #cv2.imwrite('org_labels_multiply_tgt.png',org_labels_multiply_tgt)
        number_cc_gt=numbercc(org_labels_multiply_tgt)
        if (number_cc_gt<=0):
            f.write(str(number_cc_gt)+'\n')
        else:
            f.write(str(number_cc_gt)+',')             
        
            tgt_multiply_rafi=tgt*rafi
            #cv2.imwrite('tgt_multiply_rafi.png',tgt_multiply_rafi*255)
            '''
            org_labels_multiply_tgt_multiply_rafi=org_labels*tgt_multiply_rafi
            cv2.imwrite('org_labels_multiply_tgt_multiply_rafi.png',org_labels_multiply_tgt_multiply_rafi)
            number_cc_rafi=numbercc(org_labels_multiply_tgt_multiply_rafi)
            '''
            tgt_multiply_rafi_labels = labels(tgt_multiply_rafi)
            
            number_cc_rafi_gt=0
            
            for split in (np.unique(tgt_multiply_rafi_labels)[1:]):
                tsplit=np.zeros([rows,cols],dtype=np.uint8)
                tsplit[tgt_multiply_rafi_labels==split]=1
                #cv2.imwrite('splits/tsplit'+str(split)+'.png',tsplit*255)
                org_labels_multiply_tsplit=org_labels*tsplit
                #cv2.imwrite('splits/tsplitcomps'+str(split)+'.png',org_labels_multiply_tsplit)
                number_cc_split=numbercc(org_labels_multiply_tsplit)
                if (number_cc_split<0):
                    number_cc_split=0                
                number_cc_rafi_gt=number_cc_rafi_gt+number_cc_split

            f.write(str(number_cc_rafi_gt)+',')
            r=number_cc_rafi_gt/float(number_cc_gt)
            recalls.append(r)  
            f.write(str(r)+',\n') 
    
    recall=np.mean(recalls)
    f.write('average recall= '+str(recall))
    f.close()
    print('finish'+page)
