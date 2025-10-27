# Extract features: skills, education, years experience, job titles (basic heuristics)
# Common skill keywords can be expanded
SKILL_KEYWORDS = [
    'python','sql','excel','tableau','power bi','powerbi','r','java','c++','c#',
    'machine learning','deep learning','pandas','numpy','scikit-learn','keras','tensorflow',
    'nlp','natural language processing','spark','hadoop','aws','azure','gcp','docker','kubernetes',
    'communication','leadership','management','sales','marketing'
]

def extract_skills(text, keywords=SKILL_KEYWORDS):
    found=[]
    t = text.lower()
    for k in keywords:
        if k in t:
            found.append(k)
    return list(set(found))

df['skills_list'] = df['resume_clean'].apply(lambda x: extract_skills(x))
df['num_skills'] = df['skills_list'].apply(len)

# Basic years of experience extraction - find patterns like "5 years" or "5+ years"
years_re = re.compile(r'(\b\d{1,2}\+?\s*(?:years|yrs|year)\b)')
def extract_years(text):
    m = years_re.findall(text.lower())
    if not m:
        return np.nan
    # return largest number found
    nums = []
    for match in m:
        num = re.search(r'\d{1,2}', match)
        if num:
            nums.append(int(num.group()))
    if nums:
        return max(nums)
    return np.nan

df['years_experience'] = df['resume_clean'].apply(extract_years)

# Basic education extraction
EDU_KEYWORDS = ['phd','doctor','master','m.sc','msc','mtech','mba','bachelor','b.sc','btech','b.e','bs','ba']
def extract_education(text):
    t = text.lower()
    for k in EDU_KEYWORDS:
        if k in t:
            return k
    return np.nan

df['education_level'] = df['resume_clean'].apply(extract_education)

df[['skills_list','num_skills','years_experience','education_level']].head(10)


# Wordcloud of skills / most common skills

skill_counts = Counter([s for skills in df['skills_list'] for s in skills])
skill_counts.most_common(30)
# Wordcloud
wc = WordCloud(width=800, height=400).generate_from_frequencies(skill_counts)
plt.figure(figsize=(12,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Most common skills")
plt.show()
