# Contains useful functions relating to the machine learning.
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def quick_test(pipe_param, te):
    m = fit_pipe_ms(pipe_param)
    pred = m.predict(te["Tweet description"])
    names = sorted(te['Name'].unique())
    res = classification_report(te["Name"], pred, labels=names)
    print(m.best_estimator_, "\n", res)

###### multiprocessing funcs ########

def fit_pipe_ds(pipe_param):
    tr = pipe_param[1]
    return pipe_param[0].fit(tr["Tweet description"], tr["Name"])

def pred_model_ds(model_te):
    model = model_te[0]
    te = model_te[1]
    pred = model.predict(te["Tweet description"])
    names = sorted(te['Name'].unique())
    
    res = classification_report(te["Name"], pred, labels=names)
    print(model, "\n", res)
    return accuracy_score(te["Name"], pred)

def fit_pipe_ms(pipe_param):
    tr = pipe_param[2]
    tr_label = pipe_param[3]
    gs_cv = GridSearchCV(pipe_param[0], pipe_param[1])
    return gs_cv.fit(tr, tr_label)

def pred_model_ms(model_te):
    model = model_te[0]
    te = model_te[1]
    te_lab = model_te[2]
    pred = model.predict(te)
    names = sorted(te_lab.unique())
    res = classification_report(te_lab, pred, labels=names)
    print(model.best_estimator_, "\n", res)
    print(accuracy_score(te_lab, pred))
    return model.best_estimator_

def fit_and_pred_fes(pipe_param):
    model = pipe_param[0]
    tr = pipe_param[1]
    tr_lab = pipe_param[2]
    te = pipe_param[3]
    te_lab = pipe_param[4]

    fit_model = model.fit(tr, tr_lab)
    pred = fit_model.predict(te)
    acc = accuracy_score(te_lab, pred)
    return acc