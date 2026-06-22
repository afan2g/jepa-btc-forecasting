import datetime as dt, json, boto3, lakeapi

def hr(t): print("\n"+"="*70+f"\n{t}\n"+"="*70)

sess = boto3.Session(region_name="eu-west-1")

hr("A. STS identity — whose AWS keys are these?")
try:
    ident = sess.client("sts").get_caller_identity()
    print("Account:", ident["Account"]); print("Arn    :", ident["Arn"])
except Exception as e:
    print("STS error:", repr(e))

hr("B. Raw lake-login lambda invoke (real error, no swallow)")
try:
    lc = sess.client("lambda")
    creds = sess.get_credentials()
    r = lc.invoke(FunctionName="lake-login-login", InvocationType="RequestResponse",
        Payload=json.dumps({"command":"login.login","api_key":creds.access_key,
            "table":"trades","anonymous_access":False,"user_agent":"diag"}))
    print("lambda payload:", r["Payload"].read().decode()[:300])
except Exception as e:
    print("lambda error:", repr(e)[:300])

hr("C. ANONYMOUS sample data — does the toolchain work at all?")
try:
    lakeapi.use_sample_data(anonymous_access=True)
    df = lakeapi.load_data(table="trades", start=None, end=None,
                           symbols=["BTC-USDT"], exchanges=["BINANCE"])
    print("SAMPLE trades rows:", len(df), "cols:", list(df.columns))
    print(df.head(3).to_string())
    print("date span:", df.filter(regex="time").min().min(), "->",
          df.filter(regex="time").max().max())
except Exception as e:
    print("sample error:", repr(e)[:300])
