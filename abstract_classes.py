#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 2017/1/9 10:14  
# @Author  : dongyouyuan  
# @File    : dyy_test.py  
#  
# 此脚本仅为一个演示脚本
from abc import abstractmethod, ABCMeta
class __net__(metaclass=ABCMeta):
	pass

class __trainable__(metaclass=ABCMeta):
	@abstractmethod
	def train(self):
		pass

class __static__(metaclass=ABCMeta):
	pass

class __dymanic__(metaclass=ABCMeta):
	pass
