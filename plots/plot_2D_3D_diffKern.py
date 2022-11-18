from sklearn.neural_network import MLPClassifier # Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split, cross_val_score # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from os.path import join as pjoin
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, plot_confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib as mpl
from sklearn.inspection import permutation_importance
#import shap

###### Feature, lablel and coordinate preparation for 2D prediction #############################################################
def feat_lbl_coor2d(dirdat, dircoor, clf, test_size):
    cor = np.load(dircoor)
    coor = cor["X"]
    Dat = np.load(dirdat)
    X, Y = Dat["X"], Dat["Y"]
    X = MinMaxScaler().fit_transform(X)
    #X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X)
    Y = pd.Series(Y)
    over = BorderlineSMOTE(random_state=46, k_neighbors=1)
    input_sm, output_sm = over.fit_resample(X, Y)
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(input_sm, output_sm, test_size=test_size, random_state=450)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=13422349)
    y_ind = y_test.index[:].to_numpy()
    cor_test = coor[y_ind,:]    
    ind_z = np.argsort(cor_test[:,2])
    cor_test = cor_test[ind_z,:]
    y_test = y_test.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test[ind_z]
    X_test = X_test[ind_z]
    uz, indsz = np.unique(cor_test[:,2], return_index=True)
    #scores = cross_val_score(clf, X_train_sm,y_train_sm, cv=5)
    #print(scores)
    #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std())) 
    clf = clf.fit(X_train_sm,y_train_sm)    
    y_pred = clf.predict(X_test)
    return cor_test, indsz, y_test, y_pred

###### Feature, lablel and coordinate preparation for 3D prediction #############################################################
def feat_lbl_coor3d(dirdat, dircoor, clf):
    cor = np.load(dircoor)
    coor = cor["X"]
    Dat = np.load(dirdat)
    X, Y = Dat["X"], Dat["Y"]
    X = MinMaxScaler().fit_transform(X)
    #X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X)
    Y = pd.Series(Y)
    over = BorderlineSMOTE(random_state=46, k_neighbors=1)
    input_sm, output_sm = over.fit_resample(X, Y)
    #test_size = [0.4, 0.3, 0.2, 0.1, 0.07]
    test_size = [0.3, 0.2, 0.1, 0.07]    
    target = []
    pred = []
    crd = []
    for t in test_size: 
      X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(input_sm, output_sm, test_size=t, random_state=450)
      X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=t, random_state=13422349)
      X_test = X_test.to_numpy()            
      y_ind = y_test.index[:].to_numpy()
      cor_test = coor[y_ind,:]
      y_test = y_test.to_numpy()
      target.append(y_test)
      crd.append(cor_test)
      #scores = cross_val_score(clf, X_train_sm,y_train_sm, cv=5)
      #print(scores)
      #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std())) 
      clf = clf.fit(X_train_sm,y_train_sm)
      y_pred = clf.predict(X_test)
      pred.append(y_pred)
    return target, pred, crd  

####### 2D plot of different layers in the structure #################################### 
def plot_2D(cor_test, indsz, y, nrows=5, ncols=4, N=2, figsize = [6.0, 8.0], 
            size=0.35, k=1, prt=1, mdl="knn", sec1=True, sec2=True, test=True):
    cmap = plt.cm.RdBu    
    cmaplist = [cmap(i) for i in range(cmap.N)]    
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)                           
    #cm_bright = ListedColormap(['#FFCC00', '#660066'])
    cm_bright = ListedColormap(['#DF321A', '#006cd8'])    
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)        
    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        if sec1:            
          axi.scatter(cor_test[indsz[i+150]:indsz[i+151],0],cor_test[indsz[i+150]:indsz[i+151],1],
          c=y[indsz[i+150]:indsz[i+151]],cmap=cm_bright, norm=norm)            
          # get indices of row/column
          rowid = i // ncols
          colid = i % ncols
          # write row/col indices as axes' title for identification
          axi.set_title("Layer:"+str(i+150), fontsize=20, fontweight="bold")
        elif sec2:
          axi.scatter(cor_test[indsz[i+215]:indsz[i+216],0],cor_test[indsz[i+215]:indsz[i+216],1],
          c=y[indsz[i+215]:indsz[i+216]],cmap=cm_bright, norm=norm)            
          # get indices of row/column
          rowid = i // ncols
          colid = i % ncols
          # write row/col indices as axes' title for identification
          axi.set_title("Layer:"+str(i+215), fontsize=24, fontweight="bold")
        else:
          axi.scatter(cor_test[indsz[i+200]:indsz[i+201],0],cor_test[indsz[i+200]:indsz[i+201],1],
          c=y[indsz[i+200]:indsz[i+201]],cmap=cm_bright, norm=norm)            
          # get indices of row/column
          rowid = i // ncols
          colid = i % ncols
          # write row/col indices as axes' title for identification
          axi.set_title("Layer:"+str(i+200), fontsize=20, fontweight="bold")   
    if sec2:
          fig.tight_layout()    
    if test:
        plt.savefig("GT_2d_test_revise"+str(prt)+"_"+str(k)+"x"+str(k)+"x"+str(k)+"_"+str(size)+".png", dpi=600)
    else:
        plt.savefig("pred_2d_test_revise"+mdl+"_"+str(prt)+"_"+str(k)+"x"+str(k)+"x"+str(k)+"_"+str(size)+".png", dpi=600)    

####### 3D plot of reconstruction ####################################
def plot_3D(cor_test, y_gt, y_prd, size=0.30, k=1, mdl="knn"):
    cm_bright = ListedColormap(['#FFCC00', '#660066'])
    fig = plt.figure(figsize=plt.figaspect(0.5))    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(xs = cor_test[:,0], ys = cor_test[:,1], zs = cor_test[:,2], c=y_gt, s=50, 
            facecolors='none', edgecolors='k', cmap=cm_bright, linewidths=0.2)    
    ax1.set_title("Ground Truth for "+str(size)+" of the data")
    ax1.set_xlabel("x coord")
    ax1.set_ylabel("y coord")
    ax1.set_zlabel("z coord")
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')    
    ax2.scatter(xs = cor_test[:,0], ys = cor_test[:,1], zs = cor_test[:,2], c=y_prd, s=50, 
                facecolors='none', edgecolors='k', cmap=cm_bright, linewidths=0.2)    
    ax2.set_title("Prediction for "+str(size)+" of the data"+" (test data)")
    ax2.set_xlabel("x coord")
    ax2.set_ylabel("y coord")
    ax2.set_zlabel("z coord")
    plt.savefig("GRTR_PRE_Test"+mdl+"_"+str(k)+"x"+str(k)+"x"+str(k)+"_"+str(size)+".png", dpi=300)      
"""
##### Feature Importance #############################################################
def feature_importance(dirdat, model, test_size=0.3, n=20, k="3x3x3", permute=True, bar=True):
    Dat = np.load(dirdat)
    X, Y = Dat["X"], Dat["Y"]
    X = MinMaxScaler().fit_transform(X)    
    over = BorderlineSMOTE(random_state=46, k_neighbors=1)
    input_sm, output_sm = over.fit_resample(X, Y)
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(input_sm, output_sm, test_size=test_size, random_state=450)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=13422349)        
    model.fit(X_train_sm, y_train_sm)
    if permute:    
        perm_importance = permutation_importance(model, X_test, y_test, scoring='accuracy', random_state=23)    
        importance = perm_importance.importances_mean
        ranked = np.argsort(importance)
        largest_indices = ranked[::-1][:n]
        print(largest_indices)    
        plt.barh([x for x in range(len(importance))], importance)            
        plt.savefig("permutation_feature_importance_"+str(model)+"_"+k)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        ranked = np.argsort(shap_values)
        largest_indices = ranked[::-1][:n]        
        print(largest_indices)
        if bar:
          shap.summary_plot(shap_values, X_test, plot_type="bar")
          plt.savefig("shape_feature_importance_"+str(model)+"_"+k)
        else:
          shap.summary_plot(shap_values, X_test)
          plt.savefig("shape_value_feature_importance_"+str(model)+"_"+k)
"""                  