import os
import shutil


def collect_merge_delete(dir, num=10, ifdelete=False):
    subs_dir = os.path.join(dir, 'subs')
    alls = [''] * num
    for fn in os.listdir(subs_dir):
        ffn = os.path.join(subs_dir, fn)
        with open(ffn) as f:
            lines = f.readlines()
        for l in lines:
            ll = l.strip()
            id = int(ll.split('\t')[0])
            if alls[id] != '':
                raise ValueError
            alls[id] = ll
    for idx, span in enumerate(alls):
        if span == '':
            # print(idx)
            raise ValueError
    fin_dir = os.path.join(dir, 'full_preds.txt')
    with open(fin_dir, 'w') as f:
        f.write('\n'.join(alls))

    if ifdelete:
        shutil.rmtree(subs_dir)


if __name__ == '__main__':
    for subdir in os.listdir('./outputs'):
        fsubdir = os.path.join('./outputs', subdir)
        for subsubdir in os.listdir(fsubdir):
            fsubsubdir = os.path.join(fsubdir, subsubdir)
            if 'subs' in os.listdir(fsubsubdir) and 'full_preds.txt' not in os.listdir(fsubsubdir):
                # try merge
                try:
                    collect_merge_delete(dir=fsubsubdir, ifdelete=True, num=10)
                    print('Merge success!')
                    print('Eval!')
                    print(f'task  : {subdir}')
                    print(f'paras : {subsubdir}')
                    ffn = os.path.join(fsubsubdir, 'full_preds.txt')
                    os.system(f"python squad_evaluate.py --taskname {subdir} --pred_filename {ffn}")
                    print('-----')
                except:
                    pass
