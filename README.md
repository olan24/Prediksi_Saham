# Prediksi_Haraga_Saham_Bank_Mandiri

1. DOMAIN PROYEK

Prediksi nilai saham adalah topik menarik dalam bidang pembelajaran mesin. Hal ini karena faktor-faktor yang memengaruhi harga saham tidak dapat diprediksi dengan pasti[1]. Dengan mengamati dan menganalisis permintaan dan penawaran saham, kita dapat menentukan arah pergerakan harga saham. Jika permintaan melebihi penawaran, harga saham cenderung naik, sedangkan jika permintaan lebih rendah dari penawaran, harga saham cenderung turun[2]. Ada dua jenis analisis yang umum digunakan dalam investasi saham, yaitu analisis fundamental dan teknikal. Analisis fundamental melibatkan evaluasi kondisi perusahaan baik di masa lalu maupun perkiraan di masa depan untuk memprediksi harga saham, dengan menggunakan informasi dari berita dan laporan keuangan perusahaan. Sementara itu, analisis teknikal dilakukan berdasarkan pergerakan harga saham di periode sebelumnya[3].

Penggunaan pembelajaran mesin (Machine Learning) adalah metode yang akurat dalam memprediksi harga saham. Pembelajaran mesin merujuk pada kemampuan komputer atau program untuk memproses data dan digunakan sebagai sumber informasi dalam pengambilan keputusan dan pemecahan masalah[4]. Saham dapat dikategorikan sebagai deret waktu yang tidak stabil, dan telah dikembangkan beberapa teknik untuk memprediksi pergerakan harga saham. Salah satu teknik yang banyak digunakan oleh investor adalah Support Vector Machine (SVM) untuk memprediksi pergerakan indeks harga saham[5].

2. BUSINESS UNDERSTANDING

    2.1. Problem Statement 

Dalam konteks ini, tujuan investor adalah membuat keputusan investasi yang cerdas dan terinformasi mengenai saham PT. Bank Mandiri. Investor berharap dapat menggunakan prediksi harga saham sebagai panduan untuk mengambil keputusan investasi yang tepat waktu dan menguntungkan[2]. Oleh karena itu, analisis indeks harga saham PT. Bank Mandiri menjadi penting untuk memperoleh pemahaman yang mendalam tentang tren dan perilaku pasar.

Contoh Kasus:
Untuk menggambarkan bagaimana pemanfaatan algoritma SVM (Support Vector Machine) dapat membantu investor dalam mengambil keputusan investasi pada PT. Bank Mandiri, berikut ini merupakan contoh kasusnya:

Seorang investor yang berminat untuk membeli saham PT. Bank Mandiri ingin mengetahui apakah harga saham perusahaan tersebut akan mengalami kenaikan atau penurunan dalam periode waktu tertentu. Untuk mencapai tujuan ini, investor melakukan pengumpulan data historis mengenai harga saham Bank Mandiri seperti harga open, high, low, closing, dan volume.

Setelah mengumpulkan data tersebut, investor menggunakan algoritma SVM untuk melakukan analisis terhadap indeks harga saham PT. Bank Mandiri. Algoritma SVM dipilih karena reputasinya yang terbukti dalam pemodelan dan prediksi tren pasar yang kompleks.

Investor memisahkan data menjadi dua bagian: data latihan (untuk melatih model SVM) dan data pengujian (untuk menguji keakuratan model). Data latihan berisi informasi historis mengenai harga saham Bank Mandiri, sedangkan data pengujian berisi data harga saham yang lebih baru dan belum dikenal oleh model.

Setelah melatih model SVM menggunakan data latihan, investor dapat memanfaatkan model tersebut untuk memprediksi harga saham PT. Bank Mandiri dalam periode waktu tertentu berdasarkan data pengujian. Model SVM akan memberikan prediksi apakah harga saham Bank Mandiri diperkirakan akan mengalami kenaikan atau penurunan.

Investor kemudian dapat menggunakan prediksi harga saham sebagai salah satu faktor dalam pengambilan keputusan investasi. Sebagai contoh, apabila model SVM menunjukkan bahwa harga saham Bank Mandiri diperkirakan akan mengalami kenaikan, investor mungkin akan mempertimbangkan untuk membeli saham tersebut. Di sisi lain, jika prediksi menunjukkan penurunan harga saham, investor dapat memutuskan untuk menjual atau menunda pembelian saham.

Dengan memanfaatkan algoritma SVM dan melakukan analisis terhadap indeks harga saham PT. Bank Mandiri, investor dapat memperoleh wawasan yang lebih baik mengenai tren pasar dan membuat keputusan investasi yang lebih terinformasi. Meskipun demikian, perlu diingat bahwa prediksi pasar saham tidak selalu akurat secara mutlak, sehingga keputusan investasi harus didasarkan pada informasi yang komprehensif dan pemahaman yang mendalam mengenai pasar secara keseluruhan.

    2.2 Goals(tujuan)

Tujuan dari analisis indeks harga saham PT. Bank Mandiri menggunakan algoritma SVM adalah sebagai berikut:
Membantu investor dalam mengidentifikasi peluang investasi yang menguntungkan pada saham PT. Bank Mandiri.
Memperoleh pemahaman yang lebih baik tentang tren dan perilaku pasar saham Bank Mandiri.
Mengoptimalkan keputusan investasi dengan memanfaatkan prediksi harga saham yang dihasilkan oleh model SVM.
Memaksimalkan keuntungan investor melalui pembelian dan penjualan saham pada waktu yang tepat.
Mengurangi risiko investasi dengan memperoleh wawasan yang lebih terinformasi mengenai pergerakan harga saham Bank Mandiri.
Dengan mencapai tujuan-tujuan ini, investor dapat meningkatkan kinerja investasi mereka dan mencapai hasil yang lebih baik dalam pasar saham PT. Bank Mandiri.

3. DATA UNDERSTANDING

Data pada project ini menggunakan dataset dari Kaggle.com (https://www.kaggle.com/datasets/muamkh/ihsgstockdata) 

    3.1 Overview Data
- Nama dataset : Data stock daily diambil dari tanggal 16 April 2001 sampai dengan 6 Januari 2023.

Jadi, berdasarkan informasi data daily harga saham bank mandiri yaitu 5085 data dengan 6 label yaitu: timestamp, open, low, high, close, dan volume.

    3.2 Analisis Deskriptif

Dalam menganalisis, project ini menggunakan metode EDA.

- Jumlah data adalah 5085
- Rata-rata data berada di angka 3980 – 4084 dari keseluruhan data
- Nilai terendah data berada di angka 368 pada harga close saham (Rp) 
- Nilai tertinggi data pada harga close saham yaitu 10900 dalam bentuk rupiah (Rp)
- mengubah kolom 'timestamp' menjadi tipe data datetime, yang berguna untuk analisis waktu dan pengindeksan berdasarkan waktu dalam DataFrame.

Jadi, 
- Ukuran DataFrame: DataFrame memiliki 5085 baris dan 6 kolom, menunjukkan bahwa data mencakup 5085 entri atau pengamatan yang berbeda.
- Kolom-kolom: DataFrame terdiri dari 6 kolom dengan label sebagai berikut:
- timestamp: Kolom ini berisi nilai tanggal dan waktu dalam format datetime64[ns]. Hal ini memungkinkan penggunaan fungsi dan metode yang disediakan oleh tipe data datetime.
open, low, high, close: Kolom-kolom ini berisi nilai bilangan bulat yang mewakili harga saham dalam rentang tertentu, yaitu nilai pembukaan ( open), terendah ( low), tertinggi ( high), dan penutupan ( close) pada setiap entri data.
volume: Kolom ini mengandung nilai integer yang mewakili volume perdagangan saham pada setiap entri data.
- Tipe data: Sebagian besar kolom dalam DataFrame memiliki tipe data int64, menunjukkan bahwa nilai-nilai dalam kolom tersebut adalah bilangan bulat. Kolom tanggal timestampmemiliki tipe data datetime64[ns], yang menyimpan dan waktu dalam format yang dapat diolah.
- Non-Null Count: Setiap kolom dalam DataFrame memiliki 5085 non-null count, yang menunjukkan bahwa tidak ada nilai yang hilang (null) dalam dataset. Hal ini berarti tidak ada data entri yang kosong atau tidak terisi.
- Penggunaan Memori: DataFrame df_dailymenggunakan memori sekitar 238.5 KB untuk menyimpan data dalam format yang sesuai dengan kolom tipe data.

Informasi di atas memberikan gambaran tentang struktur dan jenis data yang terdapat dalam DataFrame. Hal ini penting dalam analisis data lebih lanjut, seperti visualisasi, pemodelan, atau penarikan kesimpulan berdasarkan dataset tersebut.

    3.3 visualisasi Data 
menampilkan visualisasi distribusi dataframe dengan histogram

4. DATA PREPARATION

Untuk mengetahui sebaran distribusi data kecenderungan pusat, serta adanya nilai ekstrem atau outlier dalam setiap fitur maka perlu dibuatkan sebuah plot sebagai gambaran. tahap ini dilakukan pengelompokkan data harian berdasarkan tahun dan menghitung rata-rata nilai pada kolom-kolom 'open', 'high', 'low', dan 'close'. Selanjutnya, membuat subplot dengan ukuran (20, 10) untuk menampilkan 4 grafik bar terpisah, masing-masing untuk kolom-kolom tersebut.
jadi, Berdasarkan Nilai kuartal akhir Bank Mandiri setelah melakukan pengumuman kuartal akhir mengalami penurunan pada harga close. Begitupun juga yang terjadi pada volume turut mengalami penurunan.

dari matrix correlation diatas, dapat diketahui:

- Korelasi antara timestamp dan year sangat tinggi (0.998682). Ini menunjukkan bahwa fitur timestamp dan year memiliki korelasi yang kuat dan hampir identik. Karena itu, mungkin ada redundansi informasi antara kedua fitur tersebut.

- Korelasi antara timestamp dan low-high adalah -0.646914, yang menunjukkan adanya korelasi negatif yang kuat antara timestamp dan perbedaan nilai low-high. Hal ini bisa menunjukkan bahwa dalam periode tertentu, jika timestamp semakin tinggi, perbedaan nilai antara low dan high semakin rendah.

- Korelasi antara month dan is_quarter_end adalah 0.201347. Ini menunjukkan adanya korelasi positif yang sedang antara bulan dan penanda akhir kuartal. Artinya, kemungkinan besar penanda akhir kuartal muncul pada bulan-bulan tertentu.

- Korelasi antara open-close dan target adalah 0.008508. Ini menunjukkan adanya korelasi positif yang sangat lemah antara perbedaan nilai open-close dan target. Hal ini menunjukkan bahwa perbedaan nilai antara open dan close mungkin memiliki pengaruh minimal terhadap nilai target.

- Korelasi antara low-high dan target adalah -0.038303. Ini menunjukkan adanya korelasi negatif yang lemah antara perbedaan nilai low-high dan target. Hal ini menunjukkan bahwa perbedaan nilai antara low dan high juga memiliki pengaruh minimal terhadap nilai target.

- Teknik preparation yang digunakan adalah standart scaler. StandardScaler adalah salah satu transformer yang digunakan dalam pemrosesan data serta dalam analisis data dan pemodelan statistik. StandardScaler digunakan untuk menormalkan atau menskalakan fitur-fitur numerik dalam sebuah dataset. Pemrosesan ini melakukan penskalaan fitur-fitur dengan menghilangkan rata-rata dan menskalakan varians menjadi 1.
- Split Data: Pada tahapan ini data dibagi meliputi Data Train 4576 (90%) dan Data Valid/Test 509 (10%) dari keseluruhan data.

5. MODELLING

- Random Forest (RF) diperkenalkan oleh Breiman pada tahun 2001. RF biasanya digunakan untuk permasalahan klasifikasi dan Regresi yang melibatkan dataset dalam jumlah besar. Selain itu, RF juga termasuk dalam kategori algoritma ensemble learning. Dalam kasus klasifikasi, penentuan suara terbanyak ditentukan berdasarkan vote setiap tree sedangkan dalam kasus regresi ditetapkan berdasarkan pada nilai rata-rata setiap tree[4].
- K-Nearest Neighbour adalah  KNN dilakukan dengan cara mengklasifikasikan data input berdasarkan data pembelajaran yang jarak tetangganya paling dekat atau memiliki nilai selisih yang kecil dengan data input tersebut. Jarak antara data input dengan training sample yang sudah ada dihitung dengan Euclidean Distance[3]. 
- Jaringan Syaraf Tiruan atau dalam istilah internasionalnya Neural Networks bermaksud meniru cara kerja otak makhluk hidup. Komponen utamanya dari JST antara lain neuron dan sinaps. Neuron berisi informasi suatu informasi dapat diteruskan atau tidak sedangkan sinaps berisi hubungan antara satu neuron dengan lainnya[5].

6. EVALUASI/RESULT

- Kesalahan Kuadrat Rata-Rata (MSE):
MSE adalah ukuran yang memperkirakan kesalahan kuadrat rata-rata antara nilai prediksi dan nilai sebenarnya dari kumpulan data. Semakin rendah nilai MSE, semakin baik model regresi memprediksi nilai target.
MSE dihitung dengan menjumlahkan kesalahan kuadrat (selisih antara nilai prediksi dan nilai aktual) untuk setiap titik data, lalu membaginya dengan jumlah total titik data. Rumusnya adalah sebagai berikut:

MSE = (1/n) * Σ(yi - ŷi)^2

di mana yi adalah nilai sebenarnya, ŷi adalah nilai prediksi, dan n adalah jumlah titik data.

- R-kuadrat (R2):
R2 adalah metrik evaluasi yang mengukur sejauh mana variasi variabel target dapat dijelaskan oleh model regresi. Nilai R2 berkisar dari 0 hingga 1, dengan 0 menunjukkan bahwa model tidak dapat menjelaskan variasi sama sekali dan 1 menunjukkan bahwa model dapat menjelaskan semua variasi dengan sempurna.
R2 dihitung dengan membandingkan varian prediksi model dengan varian nilai target aktual. Nilai R2 yang positif menunjukkan bahwa model membuat prediksi yang lebih baik daripada prediksi yang hanya menggunakan rata-rata. Rumusnya adalah sebagai berikut:

R2 = 1 - (SSR/SST)

di mana SSR adalah jumlah residu kuadrat dan SST adalah jumlah total kuadrat.
Secara umum, semakin tinggi nilai R2, semakin baik model regresi dalam menjelaskan variasi data. Namun, R2 juga memiliki keterbatasan dan harus dianalisis bersama dengan metrik lainnya. 

- RMSE 
RMSE adalah akar kuadrat dari MSE. RMSE cenderung memungkinkan interpretasi yang lebih intuitif karena memiliki unit yang sama dengan variabel target asli. Seperti halnya MSE, semakin rendah nilai RMSE, semakin baik model regresi memprediksi nilai target.
Rumus RMSE:

RMSE = √MSE 
Pada hasil evaluasi modelling data ini menunjukkan MSE, R-Square (R2), dan RMSE adalah: 

artinya, berdasarkan hasil perhitumgan di atas maka Estimasi model menggunakan MSE (Mean Squared Error), R-squared (R2), dan RMSE (Root Mean Squared Error) pada tiga jenis algoritma berikut:
KNN (K-Nearest Neighbor Regressor), RF (Random Forest Regressor) dan NN (Neural Network).

* KNN (K-Nearest Neighbor Regressor):

a.	Nilai MSE (Mean Squared Error) adalah 0.2851081. MSE mengukur root mean square error antara nilai prediksi dan nilai aktual dalam kumpulan data. Semakin rendah nilai MSE, semakin baik kinerja model. Dalam hal ini, nilai MSE yang relatif tinggi menunjukkan bahwa regressor JST memiliki tingkat kesalahan yang cukup besar dalam memprediksi nilai target. Nilai R2 (R kuadrat) adalah -0.1931589. 
b.	Nilai R2 berkisar dari -∞ hingga 1, di mana 0 menunjukkan bahwa model tidak dapat menjelaskan semua variasi dan 1 menunjukkan bahwa model dapat menjelaskan semua variasi dengan sempurna. Dalam hal ini, nilai R2 negatif menunjukkan bahwa model regresi JST tidak dapat menjelaskan variasi data dengan baik.
c.	Nilai RMSE (Root Mean Squared Error) adalah 0,5339551. RMSE adalah akar kuadrat dari MSE dan menawarkan interpretasi yang lebih intuitif karena memiliki satuan yang sama dengan variabel target awal. Nilai RMSE yang lebih tinggi berarti persentase kesalahan yang lebih tinggi dalam prediksi. Dalam hal ini, nilai RMSE yang relatif tinggi menunjukkan tingkat kesalahan yang cukup besar dalam memprediksi regressor JST.

* RF (Random Forest Regressor):

a.	Nilai MSE adalah 0.2749795. Skor MSE yang lebih rendah dibandingkan dengan regressor JST menunjukkan tingkat kesalahan yang lebih rendah dalam memprediksi regressor RF.
b.	Nilai R2 adalah -0.1507715. Nilai R2 negatif menunjukkan bahwa model regresi RF juga tidak mampu menjelaskan variasi data.
c.	Nilai RMSE adalah 0,5243849. Nilai RMSE yang lebih rendah dibandingkan dengan regressor JST menunjukkan tingkat kesalahan yang lebih rendah dalam memprediksi regressor RF.

* NN (Neural Network):

a.	Nilai MSE adalah 0,2383078. Nilai MSE yang lebih rendah dibandingkan dengan regressor JST dan regressor RF menunjukkan tingkat kesalahan yang lebih rendah dalam prediksi jaringan saraf. 
b.	Nilai R2 adalah 0,0026972. Nilai R2 mendekati nol menunjukkan bahwa model jaringan saraf memberikan sedikit penjelasan untuk variabilitas data.
c.	Nilai RMSE adalah 0,4881678. Nilai RMSE yang lebih rendah dibandingkan dengan regressor JST dan regressor RF menunjukkan tingkat kesalahan yang lebih rendah dalam prediksi jaringan saraf.

7. KESIMPULAN

Dalam penelitian ini dapat disimpulkan sebagai berikut:

- KNN (K-Nearest Neighbour Regressor) memiliki nilai MSE sebesar 0.2851081, nilai R-squared sebesar -0.1931589, dan nilai RMSE sebesar 0.5339551. Model KNN Regressor memiliki tingkat kesalahan yang cukup besar dalam memprediksi nilai target, dan kemampuannya untuk menjelaskan variasi dalam data sangat rendah.

- RF (Random Forest Regressor) memiliki nilai MSE sebesar 0.2749795, nilai R-squared sebesar -0.1507715, dan nilai RMSE sebesar 0.5243849. Model RF Regressor memiliki tingkat kesalahan yang sedikit lebih kecil dibandingkan dengan KNN Regressor, tetapi tetap memiliki kinerja yang rendah dalam menjelaskan variasi dalam data.

- NN (Neural Network) memiliki nilai MSE sebesar 0.2383078, nilai R-squared sebesar 0.0026972, dan nilai RMSE sebesar 0.4881678. Model Neural Network memiliki tingkat kesalahan yang relatif lebih kecil dibandingkan dengan KNN Regressor dan RF Regressor, dan kemampuannya dalam menjelaskan variasi dalam data sedikit lebih baik.

Dari ketiga algoritma tersebut, Neural Network menunjukkan performa yang paling baik dengan MSE dan RMSE yang lebih rendah daripada KNN Regressor dan RF Regressor. Namun, nilai R-squared untuk semua algoritma tersebut masih rendah atau negatif, menunjukkan bahwa model-model tersebut tidak mampu menjelaskan variasi dalam data dengan baik. Oleh karena itu, diperlukan peninjauan lebih lanjut dan mungkin peningkatan pada pemodelan dan fitur yang digunakan untuk mencapai hasil yang lebih baik dalam memprediksi nilai target.

8. SARAN

- Evaluasi metrik: Selain MSE, R-squared, dan RMSE, ada baiknya juga melihat metrik evaluasi lainnya seperti MAE (Mean Absolute Error) dan MAPE (Mean Absolute Percentage Error) untuk mendapatkan gambaran yang lebih komprehensif tentang performa model. Metrik evaluasi tambahan ini dapat memberikan wawasan yang lebih lengkap tentang kesalahan model dalam memprediksi nilai target.

- Penyempurnaan model: Mengingat performa yang rendah dalam menjelaskan variasi dalam data, disarankan untuk melakukan penyempurnaan pada model-model yang digunakan. Mungkin perlu mengubah konfigurasi atau parameter pada algoritma KNN, RF, dan NN untuk meningkatkan performa prediksi. Eksperimen dengan berbagai parameter dan teknik tuning dapat membantu meningkatkan hasil prediksi.

- Pemilihan fitur: Perlu dipertimbangkan pemilihan fitur yang lebih relevan dan informatif untuk meningkatkan performa model. Evaluasi ulang fitur yang digunakan dalam model dapat membantu dalam memilih fitur yang lebih penting dan memiliki hubungan yang lebih kuat dengan variabel target.

- Data tambahan: Jika memungkinkan, penambahan data tambahan atau pengumpulan data yang lebih lengkap dan representatif dapat meningkatkan performa model. Dengan memiliki lebih banyak data, model dapat menemukan pola yang lebih baik dan memberikan prediksi yang lebih akurat.

- Model alternatif: Selain algoritma yang telah digunakan, ada baiknya juga mengevaluasi model alternatif. Mungkin ada algoritma lain yang lebih cocok atau memiliki performa yang lebih baik dalam menyelesaikan masalah ini. Mengeksplorasi model lain seperti regresi linear, decision tree, atau ensemble model lainnya dapat memberikan pemahaman yang lebih baik tentang mana model yang paling sesuai untuk dataset ini.

- Validasi ulang: Melakukan validasi ulang terhadap model yang telah ditingkatkan dan melakukan perbandingan dengan model-model alternatif. Validasi silang (cross-validation) atau penggunaan dataset validasi yang lebih besar dapat memberikan kepercayaan yang lebih tinggi terhadap performa model.

Dengan melakukan penyempurnaan pada pemodelan, pemilihan fitur yang tepat, penambahan data, dan evaluasi model alternatif, diharapkan dapat mencapai hasil yang lebih baik dalam memprediksi nilai target dan meningkatkan kemampuan model dalam menjelaskan variasi dalam data.

9. DAFTAR ISI

[1] K. M. Hindrayani, I. G. S. Mas Diyasa, P. A. Riyantoko, dan T. M. Fahrudin, “Studi Literatur Mengenai Prediksi Harga Saham Menggunakan Machine Learning,” Pros. Semin. Nas. Inform. Bela Negara, vol. 1, hal. 71–75, 2020, doi: 10.33005/santika.v1i0.20.

[2] R. H. Kusumodestoni dan S. Sarwido, “Komparasi Model Support Vector Machines (Svm) Dan Neural Network Untuk Mengetahui Tingkat Akurasi Prediksi Tertinggi Harga Saham,” J. Inform. Upgris, vol. 3, no. 1, 2017, doi: 10.26877/jiu.v3i1.1536.

[3] A. B. Untoro, “Prediksi Harga Saham Dengan Menggunakan Jaringan Syaraf Tiruan”, Jurnal Teknologi Informatika dan Komputer MH Thamrin, vol. 6, no. 2, pp.103-111, 2020.

[4] Patriya, “Implementasi Support Vector Machine Pada Prediksi Harga Saham Gabungan (IHSG),” Jurnal Ilmiah Teknologi dan Rekayasa, vol. 25, no. 1, pp. 24–38, 2020.

[5] F. R. Setiawan, R. F. Umbara, and A.A. Rohmawati, “Prediksi Pergerakan Harga Saham dengan Metode Support Vector Machine (SVM) Menggunakan Trend Deterministic Data Preparation”, e-Proxeeding of Engineering, vol. 5m no. 3, pp.8356-8372, 2018.

[6] R. Akmalia, I. Slamet, dan H. Pratiwi, “Analisis Sentimen Twitter Berbahasa Indonesia Terhadap Aplikasi,” Pros. Semin. Nas. MIPA UNIPA, vol. 2022, hal. 150–156, 2022.

[7] R. Primartha, Algoritma Machine Learning. Informatika, 2021.

[8] R. T. Handayanto dan Herlawati, Data Mining dan Machine Learning Menggunakan Matlab & Python. Penerbit informatika, 2020.

[5] R. T. Handayanto dan Herlawati, Data Mining dan Machine Learning Menggunakan Matlab & Python. Penerbit informatika, 2020.
