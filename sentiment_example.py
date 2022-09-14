import numpy as np
np.random.seed(0)  # نقوم بتثبيت جذر نموذج الأرقام العشوائية، حتى نحصل على نفس النتائج كل مرة نشغل البرنامج
import glob  # نستدعي مكتبة استخراج أسماء الملفات داخل مجلد

# باستخدام مكتبة glob نستخلص أسماء الملفات الموجودة داخل كل مجلد
# لاحظ أننا نريد الحصول على جميع الملفات من نوع txt لذا قمنا بوضع نجمة متبوعة ب.txt
negative_files = glob.glob('sentiment_data/Negative/*.txt')
positive_files = glob.glob('sentiment_data/Positive/*.txt')

print("ملفات النصوص الإيجابية هي:")
print(negative_files)
print("ملفات النصوص السلبية هي:")
print(positive_files)


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

# ننشئ قائمتين إضافيتين للتصنيفات
# سنرمز للتصنيف الٌيجابي بصفر و التصنيف السلبي بواحد
''' هكذا سيكون تصنيف النص الإيجابي رقم س في قائمة النصوص الإٌيجابية موجودًا في المكان رقم س أيضا في قاذمة تصنيفات النصوص 
الإيجابية، و نفس الشيء بالنسبة للنصوص السلبية'''

positive_labels = [1]*len(positive_texts)  # قائمة تصنيفات النصوص الإيجابية
negative_labels = [0]*len(negative_texts)  # قائمة تصنيفات النصوص السلبية

all_texts = positive_texts + negative_texts  # نضع جميع النصوص في قائمة واحدة
all_labels = positive_labels + negative_labels  # نضع التصنيفات في قائمة واحدة بنفس الترتيب

print("عدد النصوص يساوي عدد التصنيفات؟")
print(len(all_labels) == len(all_texts))  # لابد أن يكون لهما نفس العدد حيث يكون لكل نص تصنيف

# لاحظ أن النصوص الإيجابية جميعها في البداية و السلبية جميعها في النهاية
# لذا سنقوم بتغيير ترتيب النصوص بشكل عشوائي (مع تصنيفاتها)
# هذه الدالة من مكتبة إس كي ليرن ستقوم بتغيير أماكن النصوص عشوائيا مع مراعاة ارتباط كل نص بتصنيفه
# يمكن إدخال أي عدد من القوائم أو المصفوفات بشرط أن يكون لها نفس الطول
from sklearn.utils import shuffle  # نستدعي الدالة
all_texts, all_labels = shuffle(all_texts, all_labels)  # نشغلها، يجب مراعاة الترتيب للحصول على نتائج صحيحة

# سنكون بحاجة إلى لمعرفة مدى جودة أداء النموذج بعد تدريبه، أي سنحتاج لمجموعة مصنفة من النصوص لنختبر النموذج
# لذا نقسم البيانات لقسمين، قسم للاختبار و قسم للتدريب
# يمثل قسم التدريب ٢٠٪ من حجم النصوص الآصلي
# سنسخدم دالة train_test_split من إس كي ليرن، تقوم هذه الدالة بتقسيم النصوص و التصنيفات
# يمكن إدخال أي عدد من القوائم أو المصفوفات إلى هذه الدالة
# لاحظ أنك لو أدخلت س من القوائم/المصفوفات فإنك ستحصل على ٢*س من القوائم/المصفوفات
from sklearn.model_selection import train_test_split  # نستدعي الدالة
x_train, x_test, y_train, y_test = train_test_split(all_texts, all_labels, test_size=0.20)  # نشغلها، لاحظ الترتيب

# النص الآن جائز للتدريب، ما عدا ￿أن شكله الطبيعي غير مقبول بالنسبة للنموذج، لذا سنقوم باستخراج الميزات
# كما ذكرنا سابقًا، أحد أبرز الطرق لاستخراج الميزات هي تحوسل الكلمة لرقم count vectorization
# سنسخدم نموذج CountVecotrizer من مكتبة sklearn لتحويل الكلمات ￿لأرقام، و هكذا سنكون قد استخرجنا الميزات
# يوفر هذا النموج أيضا بعض طرق اختيار الميزات، كحذف الكلمات النادرة مثلا
# لكننا سنستخدم جميع الميزات بسبب قلة بيانات التدريب
from sklearn.feature_extraction.text import CountVectorizer  # نستدعي count vectorizer
vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')  # نقوم بتعريفه، نرغب هنا بأن يستخدم الكلمات كميزات
# نقوم بتدريبه على نصوص التدريب، هذه الخطوة ستجعله يستخرج الكلمات و يرقمها كميزات
vectorizer.fit(x_train)
# ثم نحول نصوص التدريب باستخدامه، هذه الخطوة ستجعله ينشئ مصفوفة أعمدتها الميزات و صفوفها النصوص
# سيملأ كل صف بعدد المرات التي ذكرت فيها الميزة/الكلمة  في اعمود المترتبط في النص المقصود
x_train = vectorizer.transform(x_train)

# لدينا الآن بيانات جاهزة للتدريب، سنحتاج خوارزمية لنستخدمها في تدريب النموذج
# ذكرت الخريطة الارشادية سابقا أن خوارزميتي LinearSVC و Naive Bayes يوصى بهما للاستخدام مع النصوص
# سنبدأ بتجربة إس في إم
from sklearn.svm import LinearSVC  # نستدعيها
model = LinearSVC()  # نعرف النمذوج باستخدام نموذج خوارزمية إس في إم
model.fit(x_train, y_train)  # نقوم بتدريب النموذج، هذه الخطوة قد تتطلب بعض الوقت عند التشغيل بحسب حجم البيانات

# نحتاج ￿لمعرفة مدى جودة التدريب، لذا سنقوم باختبار النموذج على نصوص التدريب
# سنستخدم دالة تقييم لتحسب لنا نسبة صحة إجابات النموذج مقارنة بالتصنيفات الحقيقة
from sklearn.metrics import accuracy_score  # نستدعي دالة التقييم
# نحول نصوص الاختبار إلي أرقام باستخدام count vectorizer الذي تم بناؤه على نصوص التدريب سابقا
x_test = vectorizer.transform(x_test)
# ثم نسخدم النموذج ليتنبأ بتصنيفات النصوص، لاحظ أننا نحتفظ بالتصنيفات الحقيقة أعلاه في المتغير y_test
predictions = model.predict(x_test)
print("نسبة الصحة باستخدام خوارزمية إس في إم:")
print(accuracy_score(y_test, predictions))  # نقيم أداء النموذج بإرسال التصنيفات الحقيقية و تصنيفات النموذج
# حصلنا على نتيجة ٨١٪، أي أن النموذج استطاع أن يتنبأ بشكل صحيح على ٨١٪ من بيانات الاختبار
# تعتبر هذه النتيجة مقبولة نوعًا ما، لكن يمكننا تجربة خوارزمية أخرى

# سنجرب استخدام خوارزمية naive bayes
from sklearn.naive_bayes import MultinomialNB  # نستدعي نموذج خوارزمية naive bayes
model = MultinomialNB()  # نعيد تعريف النموذج باستخدام هذه الخوارزمية
model.fit(x_train, y_train)  # نقوم بتدريب النموذج

# الآن نقوم باختبار النموذج مرة أخرى
predictions = model.predict(x_test)
print("نسبة الصحة باستخدام خوارزمية نايف بيز:")
print(accuracy_score(y_test, predictions))
# حصلنا على نتيجة ٨٥٪ و هي أفضل من نتيجة الخوارزمية السابقة


# قد ترغب بحفظ النموذج لاستخدامه لاحقًا
# تستخدم مكتبة بيكل pickle لحفظ أي متغير
# يشترط في هذه المكتبة أن يتم حفظ المتغير و تحميله باستخدام نفس النسخة من المكتبات
# سنقوم بحفظ النموذج باستخدامها
# لاحظ أن النموذج لن يقبل أي شكل من المصفوفات غير الذي تدرب عليه، لذا يتوجب استخدام نفس الvectorizer
# لذا سنقوم بحفظ نفس الvectorizer الذي تم بناوه على بيانات التدريب
import pickle
# نقوم بفتح ملف بالاسم المطلوب للنموذج و نخصص الرمز wb للكتابة
with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)  # تم الحفظ
# نكرر الأمر للvectorizer باسم ملف مختلف طبعًا
# نقوم بفتح ملف بالاسم المطلوب للvectorizer و نخصص الرمز wb للكتابة
with open('vectorizer.pickle', 'wb') as file:
    pickle.dump(vectorizer, file)  # تم الحفظ

# لتحميل النموذج و الvectorizer بعد الحفظ نستخدم نفس المكتبة
# نقوم بفتح ملف النموذج بوضع القراءة rb
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)
# نكرر الأمر لملف الvectorizer
with open('vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)

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