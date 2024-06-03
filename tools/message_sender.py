import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr, formataddr

def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))

def message_sender(reciver_addr, subject, content):
    smtpObj = smtplib.SMTP('smtp.qq.com',587)
    smtpObj.ehlo()
    smtpObj.starttls()

    email_config = open('email_config.json', 'r')
    email_addr = email_config['email_addr']
    secret_key = email_config['secret_key']
    smtpObj.login(email_addr, secret_key)

    from_addr = email_addr
    to_addr = reciver_addr

    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = _format_addr('Pony <%s>' % from_addr)
    message['To'] =  _format_addr('Pony2 <%s>' % to_addr)

    message['Subject'] = Header(subject, 'utf-8')

    smtpObj.sendmail(from_addr, [to_addr], message.as_string())

    smtpObj.quit()

if __name__ == '__main__':
    message_sender('ponymhc@163.com', 'test', 'test message')
