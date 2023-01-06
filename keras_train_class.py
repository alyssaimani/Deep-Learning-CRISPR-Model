# model = Sequential()
# model.add(Embedding(input_dim=16, output_dim=256, input_length=23))
# model = select_model(model, "GRU_model")
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# model.fit(x = dataset_x_train, y=dataset_y_train, epochs = EPOCHS, verbose=1, batch_size = BATCH_SIZE, shuffle=True)

#for regression
# dataset_y_pred = model_reg_1.predict(dataset_x_test)
# dataset_y_pred = dataset_y_pred.reshape((-1,))
# print("pred:", np.shape(dataset_y_pred))
# print("test:", np.shape(dataset_y_test))
# res_1 = {'ytest': dataset_y_test, 'ypred': dataset_y_pred}
# df_res_1 = pd.DataFrame(res_1)
# print("Pearson Correlation Coefficient: ", df_res_1.corr('pearson'))
# print("Spearman Correlation Coefficient: ", df_res_1.corr('spearman'))

# label = ["biLSTM model", "LSTM model", "GRU model", "biGRU model", "noRNN model"]
# ROC_AUC_score = [0.84,0.72,0.73,0.86,0.74]

# ax = plt.axes()
# ax.set_facecolor("#F8F4EA")
# plt.bar(label, ROC_AUC_score, color=["#144272","#1D5997","#256DB8","#2D85E1","#4099F5"],edgecolor=("#0F3257"),linewidth=2, width=0.5)
# plt.xlabel("Model")
# plt.ylabel("ROC AUC")
# plt.title("ROC AUC Score")
# plt.savefig("roc_auc.jpg")
# plt.show()