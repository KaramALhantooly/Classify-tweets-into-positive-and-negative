import numpy as np
import glob

np.random.seed(0)
negative_files= glob.glob('sentiment_data/Negative/*.txt')
positive_files= glob.glob('sentiment_data/Positive/*.txt')

print(positive_files)
print(negative_files)


def clean_text(text):
    '''
    يجري هنا تجهيز النص من خلال مسح الرموز و محاولة تبسيطه
    :param text: النص المدخل من نوع str
    :return: النص بعد تجهيزه
    '''

    from re import sub
    text = sub('[^ةجحخهعغفقثصضشسيىبلاآتنمكوؤرإأزدءذئطظ]', ' ', text)
    text = sub(' +', ' ', text)
    text = sub('[آإأ]', 'ا', text)
    text = sub('ة', 'ه', text)

    return text

# الآن سنقوم بقراءة النصوص و حفظها في قائمتين
# نقوم بإنشاء قائمتين لملئها بالنصوص
negative_texts = []  # السلبية
positive_texts = []  # الإيجابية

# قراءة النصوص الإيجابية
for file in positive_files:
    with open(file, 'r', encoding='utf-8') as file_to_read:
        try:
            text = file_to_read.read()  # نقرأ النص
            text = clean_text(text)  # نستخدم دالة التنظيف لتنظيف و تبسيط النص
            if text == "":
                continue  # تجاهل النصوص التي تصبح فارغة بعد تنظيفها
            print(text)
            positive_texts.append(text)  # نضيفه للقائمة
            print("-" * 10)
        except UnicodeDecodeError:  # قد نحصل على هذا الخطأ بسبب الملفات التالفة
            continue  # تجاهل الملفات التالفة


# قراءة النصوص السلبية
for file in negative_files:
    with open(file, 'r', encoding='utf-8') as file_to_read:
        try:
            text = file_to_read.read()  # نقرأ النص
            text = clean_text(text)  # نستخدم دالة التنظيف لتنظيف و تبسيط النص
            if text == "":
                continue  # تجاهل النصوص التي تصبح فارغة بعد تنظيفها
            print(text)
            negative_texts.append(text)  # نضيفه للقائمة
            print("-" * 10)
        except UnicodeDecodeError:  # قد نحصل على هذا الخطأ بسبب الملفات التالفة
            continue  # تجاهل الملفات التالفة



print("عدد النصوص الإيجابية:")
print(len(positive_texts))
print("عدد النصوص السلبية:")
print(len(negative_texts))


positive_labels = [1]*len(positive_texts)  # قائمة تصنيفات النصوص الإيجابية
negative_labels = [0]*len(negative_texts)  # قائمة تصنيفات النصوص السلبية

all_texts = positive_texts + negative_texts  # نضع جميع النصوص في قائمة واحدة
all_labels = positive_labels + negative_labels  # نضع التصنيفات في قائمة واحدة بنفس الترتيب

print("عدد النصوص يساوي عدد التصنيفات؟")
print(len(all_labels) == len(all_texts))  # لابد أن يكون لهما نفس العدد حيث يكون لكل نص تصنيف


from sklearn.utils import shuffle
all_texts, all_labels = shuffle(all_texts, all_labels)

from sklearn.model_selection import train_test_split  # نستدعي الدالة
x_train, x_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.20)  # نشغلها، لاحظ الترتيب


from sklearn.feature_extraction.text import CountVectorizer  # نستدعي count vectorizer
vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')


vectorizer.fit(x_train)

x_train = vectorizer.transform(x_train)


from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(x_train, y_train)


from sklearn.metrics import accuracy_score

x_test = vectorizer.transform(x_test)

predictions = model.predict(x_test)
print("نسبة الصحة باستخدام خوارزمية إس في إم:")
print(accuracy_score(y_test, predictions))



from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()  # نعيد تعريف النموذج باستخدام هذه الخوارزمية
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print("نسبة الصحة باستخدام خوارزمية نايف بيز:")
print(accuracy_score(y_test, predictions))




# سنقوم الآن باستخدام النموذج للحصول على تصنيف نص خارجي
# مثلا هذا النص
example_test = 'أنا سعيد جدا، كانت الرحلة رائعة'
# يتوجب تنظيف النص بنفس الطريقة التي تم باستخدامها تنظيف بيانات التدريب
cleaned_example_test = clean_text(example_test)
#ثم سنقوم بتحويل النص إلى أرقام باستخدام الvectorizer الذي أنشئناه سابقًا
# لاحظ أننا سنضع النص داخل مصفوفة ليقبلها، و هذا يعني أنه يمكن إرسال مجموعة من النصوص في مصفوفة كما حدث في بيانات الاختبار سابقا
example_test_vector = vectorizer.transform([cleaned_example_test])
# أخيرا ندخل المصفوفة الناتجة إلى النموذج
example_result = model.predict(example_test_vector)
print("تصنيف الجملة:", example_test)
print(example_result[0])  # سيكون العنصر الأول بطبيعة الحال كوننا لم ندخل سوى جملة واحدة