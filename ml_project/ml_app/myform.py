from django import forms
#import country_list
from django.core import validators
def age(value):
    if value<19:
        raise forms.ValidationError("The Age of Employee must be 19 <= age <= 80")
def salary(value):
    if value<1000:
        raise forms.ValidationError("The Salary of Employee must be greater the 1000 ")

class user_info(forms.Form):
    country_name = forms.CharField(max_length=15)
    Age          = forms.IntegerField(validators=[age])
    Salary       = forms.IntegerField(validators=[salary])


    """def clean_country_name(self):
        data = self.cleaned_data["country_name"]

        if str(data) in country_list:
            return(data)
        else:
            raise forms.ValidationError("enter a valied country name")
        return(data)"""
