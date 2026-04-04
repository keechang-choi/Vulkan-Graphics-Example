import lldb
import json

def std_string_summary(valobj, internal_dict):
    try:
        ptr = (
            valobj.GetChildMemberWithName("_Mypair")
            .GetChildMemberWithName("_Myval2")
            .GetChildMemberWithName("_Bx")
            .GetChildMemberWithName("_Ptr")
        )

        s = ptr.GetSummary()
        if s:
            # s = s.strip('"')      # remove outer quotes
            # s = s.replace('\\"','"')  # unescape quotes
            s = json.loads(bytes(str(s)[1:-1], "utf-8").decode("unicode_escape"))
            
            return s

        return "<string>"
    except:
        return "<exception!>"


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        'type summary add -F lldb_string_formatter.std_string_summary -x "std::basic_string<.*>"'
    )