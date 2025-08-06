from nnet_for_hist_dropout_stride_MET_FORCING_RESTORE_Corr_MN_MO_Reduced import *
import logging
import os


if __name__ == "__main__":
    config = Config()
    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []
    summary_RMSE_train = []
    summary_ME_train = []


    # load data to memory
    filename = 'histogram_Corn_MN_MO_metForcing_May_June_without_LW_HAND' + '.npz'
    # filename = 'histogram_all_soilweather' + '.npz'
    content = np.load(os.path.join(config.load_path , filename))
    image_all = content['output_image']
    area_all = content['output_area']
    year_all = content['output_year']
    # month_all = content['output_month']
    locations_all = content['output_locations']
    index_all = content['output_index']
    loc = content['output_index']
    yield_all = area_all
    '''
    i=0
    loc_list = []
    for l in range(0,len(loc),9):
        #print(loc[l,:])
        if str(int(loc[l,:][0]))+'_'+str(int(loc[l,:][1])) not in loc_list:
            loc_list.append(str(int(loc[l,:][0]))+'_'+str(int(loc[l,:][1])))
            i=i+1
    print(i)
    k1=np.random.choice(loc_list,len(loc_list)//4,replace=0)
    np.save('k1',list(k1))
    #locations without that are in k1
    loc_list2 = []
    for i in loc_list: 
        if i not in k1: 
            loc_list2.append(i)
    
    k2=np.random.choice(loc_list2,len(loc_list2)//3,replace=0)
    np.save('k2',list(k2))
    #locations without that are in k2
    loc_list3 = []
    for i in loc_list2: 
        if i not in k2: 
            loc_list3.append(i)

    k3=np.random.choice(loc_list3,len(loc_list3)//2,replace=0)
    np.save('k3',list(k3))
    #locations without that are in k3
    loc_list4 = []
    for i in loc_list3: 
        if i not in k3: 
            loc_list4.append(i)
    #Remainings are k4
    k4 = np.array(loc_list4)
    np.save('k4',list(k4))
    '''
    
    k1 = np.load('k1.npy')
    k2 = np.load('k2.npy')
    k3 = np.load('k3.npy')
    k4 = np.load('k4.npy')
    
    
    #month = 7
    #july: 41:50

    

    
    
    # delete broken image
    '''
    list_delete=[]
    for i in range(image_all.shape[0]):
        if np.sum(image_all[i,:,:,:])<=287:
            if year_all[i]<2019:
                list_delete.append(i)
    image_all=np.delete(image_all,list_delete,0)
    yield_all=np.delete(yield_all,list_delete,0)
    year_all = np.delete(year_all,list_delete, 0)
    locations_all = np.delete(locations_all, list_delete, 0)
    index_all = np.delete(index_all, list_delete, 0)


    # keep major counties
    list_keep=[]
    for i in range(image_all.shape[0]):
        if (index_all[i,0]==5)or(index_all[i,0]==17)or(index_all[i,0]==18)or(index_all[i,0]==19)or(index_all[i,0]==20)or(index_all[i,0]==27)or(index_all[i,0]==29)or(index_all[i,0]==31)or(index_all[i,0]==38)or(index_all[i,0]==39)or(index_all[i,0]==46):
            list_keep.append(i)
    image_all=image_all[list_keep,:,:,:]
    yield_all=yield_all[list_keep]
    year_all = year_all[list_keep]
    locations_all = locations_all[list_keep,:]
    index_all = index_all[list_keep,:]

    '''
    #validation_sets = ['k1', 'k2', 'k3', 'k4']
    validation_set = [k1, k2, k3, k4]
    for loop in range(0,4):
        a=  validation_set[loop]
        #for predict_year in range(2008,2009):
        logging.basicConfig(filename=config.save_path+'/'+'/train_for_hist_alldata_loop_corn_Flood_'+str(loop+1) + '_may_jun.log',level=logging.DEBUG)
        # # split into train and validate
        # index_train = np.nonzero(year_all < predict_year)[0]
        # index_validate = np.nonzero(year_all == predict_year)[0]
        # index_test = np.nonzero(year_all == predict_year+1)[0]

        # random choose validation set

        index_validate = np.zeros((len(loc)), dtype=bool)
        index_train = np.zeros((len(loc)), dtype=bool)
        count_v=0
        count_t =0
        for i in range(0,len(loc)):
            
            for j in range(0,len(validation_set[loop])):
        #         print(loc[i,0])
        #         print(loc[i,1])
        #         print(float(a[j][0:2]))
        #         float(a[j][3:])
                    
                if loc[i,0]== float(a[j][0:2]) and loc[i,1] == float(a[j][3:]) and int(year_all[i])!=2017 and int(year_all[i])!=2012:
                    #print("found")
                    index_validate[i] = True
                    #print(index_validate[i])
                    count_v+=1
                elif loc[i,0]!= float(a[j][0:2]) and loc[i,1] != float(a[j][3:]) and int(year_all[i])!=2017 and int(year_all[i])!=2012:
                    index_train[i] = True
                    count_t += 1

        '''
        index_train = np.logical_and(year_all != 2008, month_all==month)
        index_validate = np.logical_and(year_all == 2008, month_all==month)
        '''
        t = np.nonzero(year_all)[0]
        v = np.nonzero(year_all)[0]
        index_train = t[index_train]
        index_validate = v[index_validate]

        
        print ('train size',image_all[index_train].shape[0])
        print ('validate size',image_all[index_validate].shape[0])
        #index_train = np.nonzero(year_all != predict_year)[0]
        #index_validate = np.nonzero(year_all == predict_year)[0]
        #print ('train size',index_train.shape[0])
        #print ('validate size',index_validate.shape[0])
        logging.info('train size %d',image_all[index_train].shape[0])
        logging.info('validate size %d',image_all[index_validate].shape[0])


        # # calc train image mean (for each band), and then detract (broadcast)
        # image_mean=np.mean(image_all[index_train],(0,1,2))
        # image_all = image_all - image_mean

        image_validate=image_all[index_validate]
        yield_validate=area_all[index_validate]


        for time in range(21,22,1):
            print ("time = ", time)
            RMSE_min = 100
            RMSE_min_train = 100
            g = tf.Graph()
            with g.as_default():
                # modify config
                config = Config()
                config.H=20

                

                
                
                # Launch the graph.
                # Launch the graph.
                if loop ==0:
                    model= NeuralModel(config,'net')
                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
                    
                    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                    sess.run(tf.global_variables_initializer())
                    saver=tf.train.Saver()
                else:
                    model= NeuralModel(config,'net')
                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
                    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                    #sess.run(tf.global_variables_initializer())
                    

                #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                #     sess.run(tf.initialize_all_variables())
                #     saver=tf.train.Saver()

                    checkpoint_dir = os.path.join(config.save_path)
                    latest_filename = os.path.join(checkpoint_dir,'checkpoint')
                    
                    #saver = tf.train.import_meta_graph(os.path.join(config.save_path , str(loop)+'Corrected_CNN_model.ckpt.meta'))
                    #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir, latest_filename))  
                    #sess.run(tf.global_variables_initializer())
                    saver=tf.train.Saver()
                    try:
                        #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir, latest_filename))
                        saver.restore(sess, os.path.join(config.save_path , str(loop)+'Corrected_CNN_model.ckpt') )
                        # Restore log results
                        npzfile = np.load(os.path.join(config.save_path,str(loop)+'Corrected_result.npz') )
                        summary_train_loss = npzfile['summary_train_loss'].tolist()
                        summary_eval_loss = npzfile['summary_eval_loss'].tolist()
                        summary_RMSE = npzfile['summary_RMSE'].tolist()
                        summary_ME = npzfile['summary_ME'].tolist()
                        summary_RMSE_train = npzfile['summary_RMSE_train'].tolist()
                        summary_ME_train = npzfile['summary_ME_train'].tolist()

                        print("Model restored.")
                    except:
                        print ('No history model found')
                    #sess.run(tf.global_variables_initializer())
    
                                
                for i in range(config.train_step):
                    if i==4000:
                        config.lr/=10

                    if i==20000:
                        config.lr/=10
                   
                    # index_train_batch = np.random.choice(index_train,size=config.B)
                    #index_validate_batch = np.random.choice(index_validate, size=config.B, ,replace=0) #lets try not doing this-Rehenuma

                    index_validate_batch = index_validate
                    # try data augmentation while training
                    shift = 1
                    
                    index_train_batch_1 = np.random.choice(index_train,size=config.B+shift*2 ,replace=0)
                    index_train_batch_2 = np.random.choice(index_train,size=config.B+shift*2 ,replace=0)
                    '''
                    image_train_batch = (image_all[index_train_batch_1,:,7:7+config.H,:]+image_all[index_train_batch_1,:,7:7+config.H,:])/2 #0:config.H,:
                    yield_train_batch = (yield_all[index_train_batch_1]+yield_all[index_train_batch_1])/2
                    
                    index_train_batch_1 = index_train
                    index_train_batch_2 = index_train
                    '''
                    image_train_batch = (image_all[index_train_batch_1,:,time:time+config.H,:]+image_all[index_train_batch_1,:,time:time+config.H,:])/2 #0:config.H,:
                    yield_train_batch = (yield_all[index_train_batch_1]+yield_all[index_train_batch_1])/2

                    
                    arg_index = np.argsort(yield_train_batch)
                    yield_train_batch = yield_train_batch[arg_index][shift:-shift]
                    image_train_batch = image_train_batch[arg_index][shift:-shift]
                    #print("test")
                    _, train_loss, train_loss_reg = sess.run([model.train_op, model.loss_err, model.loss_reg], feed_dict={
                        model.x:image_train_batch,
                        model.y:yield_train_batch,
                        model.lr:config.lr,
                        model.keep_prob: config.keep_prob
                        })

                    if i%500 == 0:
                        #validate with all train set Rehenuma, not with index_validate_batch   #changed to index validate 9/23/2020
                        val_loss,val_loss_reg = sess.run([model.loss_err,model.loss_reg], feed_dict={
                            model.x: image_all[index_validate, :, time:time+config.H, :],
                            model.y: yield_all[index_validate],
                            model.keep_prob: 1
                        })

                        print ('Jul Nov Validation '+str(loop+1)+' step'+str(i),train_loss,train_loss_reg,val_loss,val_loss_reg,config.lr)
                        logging.info('%d %d step %d %f %f %f %f %f',time,loop+1,i,train_loss,train_loss_reg,val_loss,val_loss_reg,config.lr)
                    
                    if i%500 == 0:
                        # do validation
                        pred = []
                        real = []
                        k = np.load('k.npy')
                        c = np.load('c.npy')
                        MaxValue= np.load('MaxValue.npy')  
                        print ('loaded log function parameters')
                        #Inverse f(n) = 10^((f(n) -c) / k) - MaxValue
                        image_check = image_all[index_validate]
                        yield_check  = yield_all[index_validate]
                        for j in range(image_check.shape[0] // config.B):
                            real_temp = yield_check[j * config.B:(j + 1) * config.B]
                            real_temp =np.power(10, (real_temp - c)/k)-MaxValue    #Get back to original scale
                            pred_temp= sess.run(model.logits, feed_dict={
                                model.x: image_check[j * config.B:(j + 1) * config.B,:,time:time+config.H,:],
                                model.y: yield_check[j * config.B:(j + 1) * config.B],
                                model.keep_prob: 1
                                })
                            pred_temp =np.power(10, (pred_temp - c)/k)-MaxValue #Get back to original scale
                            pred.append(pred_temp)
                            real.append(real_temp)
                            
                        pred=np.concatenate(pred)
                        real=np.concatenate(real)
                        RMSE=np.sqrt(np.mean((pred-real)**2))
                        ME=np.mean(pred-real)
                        RMSE_ideal = np.sqrt(np.mean((pred-ME-real)**2))
                        arg_index = np.argsort(pred)
                        pred = pred[arg_index][50:-50]
                        real = real[arg_index][50:-50]
                        ME_part = np.mean(pred-real)

                        if RMSE<RMSE_min:
                            RMSE_min=RMSE
                           

                        # do validation on Training set
                        pred_train = []
                        real_train = []
                        image_check_train = image_all[index_train]
                        yield_check_train  = yield_all[index_train]
                        for j in range(image_check_train.shape[0] // config.B):
                            real_temp_train = yield_check_train[j * config.B:(j + 1) * config.B]
                            real_temp_train =np.power(10, (real_temp_train - c)/k)-MaxValue    #Get back to original scale
                            pred_temp_train= sess.run(model.logits, feed_dict={
                                model.x: image_check_train[j * config.B:(j + 1) * config.B,:,time:time+config.H,:],
                                model.y: yield_check_train[j * config.B:(j + 1) * config.B],
                                model.keep_prob: 1
                                })
                            pred_temp_train =np.power(10, (pred_temp_train - c)/k)-MaxValue #Get back to original scale
                            pred_train.append(pred_temp_train)
                            real_train.append(real_temp_train)
                            
                        pred_train=np.concatenate(pred_train)
                        real_train=np.concatenate(real_train)
                        RMSE_train=np.sqrt(np.mean((pred_train-real_train)**2))
                        ME_train=np.mean(pred_train-real_train)
                        RMSE_ideal_train = np.sqrt(np.mean((pred_train-ME_train-real_train)**2))
                        arg_index_train = np.argsort(pred_train)
                        pred_train = pred_train[arg_index][50:-50]
                        real_train = real_train[arg_index][50:-50]
                        ME_part_train = np.mean(pred_train-real_train)

                        if RMSE_train<RMSE_min_train:
                            RMSE_min_train=RMSE_train

                           

                        # print 'Validation set','RMSE',RMSE,'ME',ME,'RMSE_min',RMSE_min
                        # logging.info('Validation set RMSE %f ME %f RMSE_min %f',RMSE,ME,RMSE_min)
                        print ('Validation set','RMSE',RMSE,'RMSE_ideal',RMSE_ideal,'ME',ME,'ME_part',ME_part,'RMSE_min',RMSE_min)
                        logging.info('Validation set RMSE %f RMSE_ideal %f ME %f ME_part %f RMSE_min %f',RMSE,RMSE_ideal,ME,ME_part,RMSE_min)

                        print ('Training set','RMSE',RMSE_train,'RMSE_ideal',RMSE_ideal_train,'ME',ME_train,'ME_part',ME_part_train,'RMSE_min',RMSE_min_train)
                        logging.info('Training set RMSE %f RMSE_ideal %f ME %f ME_part %f RMSE_min %f',RMSE_train,RMSE_ideal_train,ME_train,ME_part_train,RMSE_min_train)

        
                        summary_train_loss.append(train_loss)
                        summary_eval_loss.append(val_loss)
                        summary_RMSE.append(RMSE)
                        summary_ME.append(ME)
                        summary_RMSE_train.append(RMSE_train)
                        summary_ME_train.append(ME_train)
				
				
                # save
                save_path = saver.save(sess, os.path.join(config.save_path, str(loop+1)+'Corrected_CNN_model.ckpt'))
                print('save in file: %s' % save_path)
                logging.info('save in file: %s' % save_path)

                # save
                save_path = saver.save(sess, os.path.join(config.save_path,str(loop+1)+'Corrected_CNN_model.ckpt'))
                print('save in file: %s' % save_path)
                logging.info('save in file: %s' % save_path)

                # save result
                pred_out = []
                real_out = []
                feature_out = []
                year_out = []
                # month_out = []
                locations_out =[]
                index_out = []

                feature,pred = sess.run(
                    [model.fc6_2,model.logits], feed_dict={
                    model.x: image_all[index_validate,:,time:time+config.H,:],
                    model.y: yield_all[index_validate],
                    model.keep_prob:1
                })

                #pred = pred * (132121.981/5000) #Get back to original scale
                pred =np.power(10, (pred - c)/k)-MaxValue  #Get back to original scale
                real = yield_all[index_validate]
                #real = real * (132121.981/5000) #Get back to original scale
                real =np.power(10, (real - c)/k)-MaxValue #Get back to original scale

                pred_out.append(pred)
                real_out.append(real)
                feature_out.append(feature)
                year_out.append(year_all[index_validate])
                # month_out.append(month_all[index_validate])
                locations_out.append(locations_all[index_validate])
                index_out.append(index_all[index_validate])
                RMSE=np.sqrt(np.mean((pred-real)**2))
                print('validation RMSE = ', RMSE )
                np.save(os.path.join(config.save_path,str(loop+1)+'validation_RMSE.npy'), RMSE)



                weight_out, b_out = sess.run(
                    [model.dense_W, model.dense_B], feed_dict={
                        model.x: image_all[0 * config.B:(0 + 1) * config.B, :, time:time+config.H, :],
                        model.y: yield_all[0 * config.B:(0 + 1) * config.B],
                        model.keep_prob: 1
                    })
                pred_out=np.concatenate(pred_out)
                real_out=np.concatenate(real_out)
                feature_out=np.concatenate(feature_out)
                year_out=np.concatenate(year_out)
                # month_out = np.concatenate(month_out)
                locations_out=np.concatenate(locations_out)
                index_out=np.concatenate(index_out)
                
                np.savez(os.path.join(config.save_path,str(loop+1)+'Corrected_result_prediction.npz'),
                    pred_out=pred_out,real_out=real_out,feature_out=feature_out,
                    year_out=year_out, locations_out=locations_out,weight_out=weight_out,b_out=b_out,index_out=index_out)
                np.savez(os.path.join(config.save_path, str(loop+1)+'Corrected_result.npz'),
                    summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                    summary_RMSE=summary_RMSE,summary_ME=summary_ME, summary_RMSE_train=summary_RMSE_train,summary_ME_train=summary_ME_train)






                #save results all             
                pred_out = []
                real_out = []
                feature_out = []
                year_out = []
                # month_out = []
                locations_out =[]
                index_out = []
                for i in range(image_all.shape[0] // config.B):
                    feature,pred = sess.run(
                        [model.fc6_2,model.logits], feed_dict={
                        model.x: image_all[i * config.B:(i + 1) * config.B,:,time:time+config.H,:],
                        model.y: yield_all[i * config.B:(i + 1) * config.B],
                        model.keep_prob:1
                    })
                    real = yield_all[i * config.B:(i + 1) * config.B]

                    pred_out.append(pred)
                    real_out.append(real)
                    feature_out.append(feature)
                    year_out.append(year_all[i * config.B:(i + 1) * config.B])
                    # month_out.append(month_all[i * config.B:(i + 1) * config.B])
                    locations_out.append(locations_all[i * config.B:(i + 1) * config.B])
                    index_out.append(index_all[i * config.B:(i + 1) * config.B])
                    # print i
                weight_out, b_out = sess.run(
                    [model.dense_W, model.dense_B], feed_dict={
                        model.x: image_all[0 * config.B:(0 + 1) * config.B, :, time:time+config.H, :],
                        model.y: yield_all[0 * config.B:(0 + 1) * config.B],
                        model.keep_prob: 1
                    })
                pred_out=np.concatenate(pred_out)
                real_out=np.concatenate(real_out)
                feature_out=np.concatenate(feature_out)
                year_out=np.concatenate(year_out)
                # month_out = np.concatenate(month_out)
                locations_out=np.concatenate(locations_out)
                index_out=np.concatenate(index_out)

                np.savez(os.path.join(config.save_path, str(loop+1)+'Corrected_result_prediction_all.npz'),
                    pred_out=pred_out,real_out=real_out,feature_out=feature_out,
                    year_out=year_out, locations_out=locations_out,weight_out=weight_out,b_out=b_out,index_out=index_out)

