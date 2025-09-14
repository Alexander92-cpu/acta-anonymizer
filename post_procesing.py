#We drop the following predictions:

#"B-NAME_STUDENT" and "I-NAME_STUDENT" predictions that start with lowercase letters.
#B-URL_PERSONAL" predictions that do not contain "tp".
#B-EMAIL" predictions that do not contain "@".
#All predictions for whitespace strings other than "\n".
#B-ID_NUM" predictions for ":" and "-".
#If the probability of "O" was less than 0.60, the label with the next highest probability of "O" was output.
#Post processing for BIO Prefixes
#O, I-[PII] -> O, B-[PII]
B-[PII], B-[PII] -> B-[PII], I-[PII]
I-[PII], B-[PII] -> I-[PII], I-[PII]
Post processing for impossible prediction pattern
e.g
if predicted NAME STUDENT is Non-camel case, change "O"
if predicted ID-NUM contains special character, change "O"( r'[!@#$&%?^+=*<>]' )
After much error analysis, we found above pattern
pay special attention to \n appearing in addresses, which should be labeled with the I label
Removed all NAME_STUDENT predictions that are not title - cased, or of length 1, or contain a digit, or through blacklist(Mr., Ms., Dr., …).
If a name is mentioned multiple times in one document and one of them is pii, mark them all as pii. (very impactful)
Drop "B-ID_NUM" predictions for ":" and "-".
The
first
one is specific
for class NAME_STUDENT.Once a substring was classified as NAME_STUDENT, all other occurrences of this substring in the document were relabeled to NAME_STUDENT.However, there were cases when a single "." was predicted as NAME_STUDENT and then propagated through the entire text.To fix that, substrings of length 1 and those that are not title cased, were relabeled to O.
